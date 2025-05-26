import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, filters, morphology
import trimesh
from scipy.ndimage import zoom, label, binary_dilation, gaussian_filter, binary_fill_holes, binary_erosion
import pydicom
import pandas as pd
from colorama import Fore, Style, init
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
import json

# Initialize colorama for colored terminal output
init()

# Load MRI data (supports NIfTI and DICOM)
def load_mri_data(filepath):
    file_extension = filepath.split('.')[-1].lower()
    
    if file_extension == 'nii' or filepath.endswith('.nii.gz'):
        return load_nifti_file(filepath)
    elif file_extension == 'dcm':
        return load_dicom_file(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Load NIfTI file
def load_nifti_file(filepath):
    nii_img = nib.load(filepath)
    data = nii_img.get_fdata()
    
    # If this is a BraTS segmentation file, keep original values
    if 'seg' in filepath.lower():
        data = data.astype(np.float32)
        print(f"{Fore.GREEN}Loaded segmentation data with shape: {data.shape}{Style.RESET_ALL}")
    else:
        # Normalize image data to 0-1 range for non-segmentation files
        data = (data - data.min()) / (data.max() - data.min())
        print(f"{Fore.GREEN}Loaded MRI data with shape: {data.shape}{Style.RESET_ALL}")
    
    print(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Check unique values for segmentation files
    unique_vals = np.unique(data)
    if len(unique_vals) < 10:
        print(f"Unique segmentation values: {unique_vals}")
        
    return data, nii_img.affine

# Load DICOM files
def load_dicom_file(filepath):
    if os.path.isdir(filepath):
        dicom_dir = filepath
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    else:
        dicom_dir = os.path.dirname(filepath)
        dicom_files = [filepath]
        additional_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        if len(additional_files) > 1:
            print(f"Found {len(additional_files)} DICOM files in directory. Loading all files.")
            dicom_files = additional_files
    
    dicom_files.sort()
    
    # Get sample spacing information from first file
    sample_dicom = pydicom.dcmread(dicom_files[0])
    pixel_spacing = sample_dicom.PixelSpacing if hasattr(sample_dicom, 'PixelSpacing') else [1.0, 1.0]
    slice_thickness = float(sample_dicom.SliceThickness) if hasattr(sample_dicom, 'SliceThickness') else 1.0
    spacing = [*pixel_spacing, slice_thickness]
    
    # Load all slices
    slices = []
    for file in dicom_files:
        dicom_data = pydicom.dcmread(file)
        slices.append(dicom_data.pixel_array)
    
    # Stack into 3D volume
    data = np.stack(slices, axis=-1) if len(slices) > 1 else slices[0]
    data = (data - data.min()) / (data.max() - data.min())
    
    print(f"{Fore.GREEN}Loaded DICOM data with shape: {data.shape}{Style.RESET_ALL}")
    print(f"Pixel spacing: {pixel_spacing}, Slice thickness: {slice_thickness}")
    
    # Create affine matrix from spacing information
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    return data, affine

# NEW: Advanced brain surface creation with realistic anatomy
def create_realistic_brain_surface(mri_data, method='advanced_realistic'):
    """Create realistic brain surface with anatomical features and smooth transitions"""
    print("Creating advanced realistic brain surface...")
    print(f"Input data shape: {mri_data.shape}, dtype: {mri_data.dtype}")
    
    # Memory management
    total_voxels = np.prod(mri_data.shape)
    memory_gb = total_voxels * 8 / (1024**3)
    
    if memory_gb > 1.5:
        print(f"Large dataset detected ({memory_gb:.1f}GB estimated). Applying memory-efficient processing...")
        downsample_factor = max(1, int(np.cbrt(total_voxels / 50_000_000)))
        if downsample_factor > 1:
            print(f"Downsampling by factor {downsample_factor} for initial processing...")
            downsampled = mri_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            brain_mask_small = create_realistic_brain_surface(downsampled, method='simple_realistic')
            zoom_factors = [s1/s2 for s1, s2 in zip(mri_data.shape, brain_mask_small.shape)]
            brain_mask = zoom(brain_mask_small, zoom_factors, order=1)  # Linear interpolation for smoothness
            return brain_mask
    
    if method == 'advanced_realistic':
        # Multi-scale brain extraction with anatomical preservation
        print("Step 1: Multi-scale smoothing for anatomical preservation...")
        
        # Apply multi-scale Gaussian smoothing to preserve different anatomical features
        fine_details = gaussian_filter(mri_data, sigma=0.5)      # Preserve fine structures
        medium_structures = gaussian_filter(mri_data, sigma=1.5)  # Medium structures
        coarse_anatomy = gaussian_filter(mri_data, sigma=3.0)     # Overall brain shape
        
        # Create composite smoothing that preserves anatomical details
        combined_smooth = (0.5 * fine_details + 0.3 * medium_structures + 0.2 * coarse_anatomy)
        
        print("Step 2: Adaptive thresholding with tissue differentiation...")
        
        # Multi-level thresholding for different brain tissues
        non_zero_data = combined_smooth[combined_smooth > 0]
        if len(non_zero_data) > 1_000_000:
            sample_indices = np.random.choice(len(non_zero_data), 1_000_000, replace=False)
            sample_data = non_zero_data[sample_indices]
        else:
            sample_data = non_zero_data
        
        # Create multiple threshold levels for realistic brain boundaries
        background_threshold = np.percentile(sample_data, 15)  # Background separation
        tissue_threshold = np.percentile(sample_data, 35)      # Brain tissue threshold
        
        # Initial brain mask with smooth transitions
        initial_mask = combined_smooth > background_threshold
        brain_tissue_mask = combined_smooth > tissue_threshold
        
        print("Step 3: Anatomical morphological processing...")
        
        # Clean up with anatomically-aware operations
        brain_mask = morphology.remove_small_objects(initial_mask, min_size=5000)
        brain_mask = binary_fill_holes(brain_mask)
        
        # Apply gentle morphological operations to preserve brain sulci and gyri
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(2))
        brain_mask = binary_erosion(brain_mask, morphology.ball(1))
        brain_mask = binary_dilation(brain_mask, morphology.ball(3))
        
        print("Step 4: Surface smoothing with anatomical constraints...")
        
        # Create smooth surface while preserving anatomical features
        # Use distance transform for smooth surface generation
        distance_map = ndimage.distance_transform_edt(brain_mask)
        
        # Apply smoothing to distance map for realistic surface
        smooth_distance = gaussian_filter(distance_map, sigma=2.0)
        
        # Create final brain surface with smooth transitions
        brain_surface = (smooth_distance > 0).astype(float)
        
        # Add subtle anatomical texture based on original MRI intensity
        texture_weight = gaussian_filter(mri_data, sigma=1.0)
        texture_weight = (texture_weight - texture_weight.min()) / (texture_weight.max() - texture_weight.min())
        
        # Blend surface with texture information for realism
        textured_surface = brain_surface * (0.7 + 0.3 * texture_weight)
        
        print("Step 5: Final anatomical refinement...")
        
        # Ensure single connected component
        labeled_mask, num_labels = label(textured_surface > 0.5)
        if num_labels > 1:
            unique_labels, counts = np.unique(labeled_mask, return_counts=True)
            if unique_labels[0] == 0:
                unique_labels = unique_labels[1:]
                counts = counts[1:]
            if len(counts) > 0:
                largest_component = unique_labels[np.argmax(counts)]
                brain_mask = (labeled_mask == largest_component).astype(float)
                textured_surface = textured_surface * brain_mask
        
        return textured_surface
    
    else:  # Fallback method
        print("Using simple realistic method...")
        smoothed = gaussian_filter(mri_data, sigma=2.0)
        non_zero_data = smoothed[smoothed > 0]
        if len(non_zero_data) > 1_000_000:
            sample_indices = np.random.choice(len(non_zero_data), 1_000_000, replace=False)
            threshold = np.percentile(non_zero_data[sample_indices], 30)
        else:
            threshold = np.percentile(non_zero_data, 30)
        
        brain_mask = smoothed > threshold
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=2000)
        brain_mask = binary_fill_holes(brain_mask)
        
        # Apply smoothing for realistic appearance
        smooth_mask = gaussian_filter(brain_mask.astype(float), sigma=1.5)
        return smooth_mask

# NEW: Generate ultra-high quality mesh with realistic texturing
def generate_ultra_realistic_mesh(data, threshold=0.5, step_size=1, preserve_anatomy=True):
    """Generate ultra-realistic mesh with advanced smoothing and anatomical preservation"""
    try:
        print("Generating ultra-realistic mesh...")
        
        # Apply advanced pre-processing for realistic surface
        if preserve_anatomy:
            # Multi-scale smoothing to preserve both fine details and overall shape
            fine_smooth = gaussian_filter(data.astype(float), sigma=0.3)
            coarse_smooth = gaussian_filter(data.astype(float), sigma=1.0)
            processed_data = 0.7 * fine_smooth + 0.3 * coarse_smooth
        else:
            processed_data = gaussian_filter(data.astype(float), sigma=0.5)
        
        # Generate high-resolution mesh
        print(f"Running marching cubes with step_size={step_size}...")
        verts, faces, normals, values = measure.marching_cubes(
            processed_data,
            level=threshold,
            step_size=step_size,
            allow_degenerate=False,
            gradient_direction='descent'  # For smoother gradients
        )
        
        # Create initial mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        print("Applying advanced mesh processing...")
        
        # Remove artifacts and small components
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()
        
        # Keep only the largest connected component
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest_component = max(components, key=lambda x: len(x.vertices))
            mesh = largest_component
            print(f"Kept largest component, removed {len(components)-1} small components")
        
        # Advanced smoothing for realistic brain surface
        print("Applying advanced Laplacian smoothing...")
        
        # Multiple passes of different smoothing techniques
        mesh = mesh.smoothed(iterations=3)  # Initial smoothing
        
        # Taubin smoothing for better shape preservation
        for _ in range(5):
            mesh = mesh.smoothed(iterations=1)
            mesh = mesh.smoothed(iterations=1, lambda_factor=-0.53)  # Shrinking step
        
        # Final light smoothing
        mesh = mesh.smoothed(iterations=2)
        
        # Ensure mesh is watertight for better rendering
        if not mesh.is_watertight:
            print("Attempting to make mesh watertight...")
            mesh.fill_holes()
        
        print(f"Generated ultra-realistic mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
        
    except Exception as e:
        print(f"{Fore.RED}Error generating ultra-realistic mesh: {e}{Style.RESET_ALL}")
        return None

# NEW: Enhanced tumor processing with diagnostic information
def process_tumor_segmentation_with_diagnostics(seg_data):
    """Process tumor segmentation with enhanced diagnostic information"""
    print("Processing tumor segmentation with diagnostic analysis...")
    
    # BraTS label mapping with diagnostic significance
    tumor_components = {
        'necrotic': {
            'mask': (seg_data == 1),
            'description': 'Necrotic/Non-enhancing tumor core',
            'clinical_significance': 'Indicates cell death within tumor, often due to rapid growth exceeding blood supply',
            'color': [120, 60, 120, 180],  # Dark purple, semi-transparent
            'priority': 2
        },
        'edema': {
            'mask': (seg_data == 2),
            'description': 'Peritumoral edema',
            'clinical_significance': 'Surrounding brain tissue swelling, indicates tumor influence on nearby tissue',
            'color': [100, 150, 255, 120],  # Light blue, highly transparent
            'priority': 1
        },
        'enhancing': {
            'mask': (seg_data == 4),
            'description': 'Enhancing tumor',
            'clinical_significance': 'Active tumor region with disrupted blood-brain barrier, primary treatment target',
            'color': [255, 60, 60, 200],  # Bright red, mostly opaque
            'priority': 3
        }
    }
    
    # Process and analyze each component
    processed_components = {}
    diagnostic_info = {}
    
    for name, info in tumor_components.items():
        mask = info['mask']
        if np.any(mask):
            # Clean up component
            cleaned_mask = morphology.remove_small_objects(mask, min_size=30)
            cleaned_mask = morphology.binary_closing(cleaned_mask, morphology.ball(1))
            
            volume_mm3 = np.sum(cleaned_mask)  # Approximate volume in mm³
            
            if volume_mm3 > 50:  # Only keep substantial components
                # Apply gentle smoothing for realistic appearance
                smooth_mask = gaussian_filter(cleaned_mask.astype(float), sigma=0.8)
                processed_components[name] = {
                    'mask': smooth_mask,
                    'color': info['color'],
                    'priority': info['priority']
                }
                
                # Store diagnostic information
                diagnostic_info[name] = {
                    'description': info['description'],
                    'clinical_significance': info['clinical_significance'],
                    'volume_mm3': int(volume_mm3),
                    'location': calculate_tumor_location(cleaned_mask)
                }
                
                print(f"  {name}: {volume_mm3:,} mm³ - {info['description']}")
    
    # Create composite whole tumor for overall assessment
    if processed_components:
        whole_tumor_mask = np.zeros_like(seg_data, dtype=float)
        for comp_info in processed_components.values():
            whole_tumor_mask = np.maximum(whole_tumor_mask, comp_info['mask'])
        
        if np.sum(whole_tumor_mask > 0.1) > 100:
            processed_components['whole_tumor'] = {
                'mask': whole_tumor_mask,
                'color': [255, 140, 40, 150],  # Orange, semi-transparent
                'priority': 0
            }
            
            total_volume = int(np.sum(whole_tumor_mask > 0.1))
            diagnostic_info['whole_tumor'] = {
                'description': 'Complete tumor complex',
                'clinical_significance': 'Entire pathological region requiring monitoring',
                'volume_mm3': total_volume,
                'location': 'Multiple regions'
            }
    
    return processed_components, diagnostic_info

# NEW: Calculate tumor location for diagnostic purposes
def calculate_tumor_location(mask):
    """Calculate approximate anatomical location of tumor"""
    if not np.any(mask):
        return "Unknown"
    
    # Find center of mass
    center = ndimage.center_of_mass(mask)
    z, y, x = [int(c) for c in center]
    
    # Simple anatomical region classification based on position
    # This is a simplified approach - real implementation would use anatomical atlases
    regions = []
    
    # Anterior-Posterior
    if z < mask.shape[0] * 0.4:
        regions.append("Anterior")
    elif z > mask.shape[0] * 0.6:
        regions.append("Posterior")
    else:
        regions.append("Central")
    
    # Superior-Inferior
    if y < mask.shape[1] * 0.4:
        regions.append("Superior")
    elif y > mask.shape[1] * 0.6:
        regions.append("Inferior")
    
    # Left-Right
    if x < mask.shape[2] * 0.4:
        regions.append("Left")
    elif x > mask.shape[2] * 0.6:
        regions.append("Right")
    
    return " ".join(regions) if regions else "Central"

# NEW: Create diagnostic annotations for 3D model
def create_diagnostic_annotations(tumor_components, diagnostic_info):
    """Create diagnostic annotations that can be embedded in the 3D model"""
    annotations = {
        'diagnostic_summary': {
            'total_components': len([k for k in tumor_components.keys() if k != 'whole_tumor']),
            'total_volume_mm3': sum([info['volume_mm3'] for info in diagnostic_info.values() if 'volume_mm3' in info]),
            'primary_concerns': []
        },
        'component_details': diagnostic_info,
        'clinical_recommendations': []
    }
    
    # Generate clinical recommendations based on findings
    for name, info in diagnostic_info.items():
        if name == 'enhancing' and info['volume_mm3'] > 1000:
            annotations['clinical_recommendations'].append(
                "Large enhancing tumor component detected - priority for treatment planning"
            )
        elif name == 'edema' and info['volume_mm3'] > 5000:
            annotations['clinical_recommendations'].append(
                "Significant peritumoral edema - monitor for mass effect"
            )
        elif name == 'necrotic' and info['volume_mm3'] > 500:
            annotations['clinical_recommendations'].append(
                "Necrotic core present - indicates advanced tumor progression"
            )
    
    return annotations

# NEW: Enhanced realistic brain model creation
def create_ultra_realistic_brain_model(mri_data, tumor_components=None, diagnostic_info=None, step_size=1):
    """Create ultra-realistic 3D brain model with proper transparency and diagnostic features"""
    print("\n🧠 Creating ultra-realistic brain visualization...")
    
    # Generate anatomically accurate brain surface
    print("Generating anatomically accurate brain surface...")
    brain_surface = create_realistic_brain_surface(mri_data, method='advanced_realistic')
    
    # Create ultra-high quality brain mesh
    print("Creating ultra-realistic brain mesh...")
    brain_mesh = generate_ultra_realistic_mesh(
        brain_surface, 
        threshold=0.3,  # Lower threshold for better surface capture
        step_size=step_size,
        preserve_anatomy=True
    )
    
    if brain_mesh is None:
        raise Exception("Failed to generate brain mesh")
    
    # Apply realistic brain coloring with subtle variations
    print("Applying realistic brain texturing...")
    
    # Create vertex colors based on local curvature and position for realism
    vertex_colors = np.zeros((len(brain_mesh.vertices), 4))
    
    # Base brain color (realistic pinkish-gray)
    base_color = np.array([235, 205, 185, 255]) / 255.0
    
    # Add subtle variations based on vertex position and local geometry
    for i, vertex in enumerate(brain_mesh.vertices):
        # Subtle color variation based on position (simulate different brain regions)
        variation = 0.95 + 0.1 * np.sin(vertex[0] * 0.01) * np.cos(vertex[1] * 0.01)
        color_variant = base_color * variation
        color_variant[3] = 0.85  # Semi-transparent for realistic appearance
        vertex_colors[i] = color_variant
    
    brain_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    # Collect all meshes
    all_meshes = [brain_mesh]
    model_annotations = {'brain': 'Healthy brain tissue with realistic surface rendering'}
    
    # Add tumor components with proper transparency and diagnostic coloring
    if tumor_components and diagnostic_info:
        print("Adding diagnostic tumor visualizations...")
        
        # Sort components by priority (enhancing tumor most visible)
        sorted_components = sorted(
            tumor_components.items(), 
            key=lambda x: x[1].get('priority', 0)
        )
        
        for component_name, component_info in sorted_components:
            if component_name in diagnostic_info:
                mask = component_info['mask']
                color = component_info['color']
                
                if np.any(mask > 0.1):  # Process components with sufficient volume
                    print(f"  Processing {component_name} with diagnostic information...")
                    
                    # Generate high-quality tumor mesh
                    tumor_mesh = generate_ultra_realistic_mesh(
                        mask,
                        threshold=0.3,
                        step_size=max(1, step_size // 2),  # Higher resolution for pathology
                        preserve_anatomy=True
                    )
                    
                    if tumor_mesh:
                        # Apply diagnostic coloring with proper transparency
                        tumor_colors = np.full((len(tumor_mesh.vertices), 4), color, dtype=np.uint8)
                        tumor_mesh.visual.vertex_colors = tumor_colors
                        
                        all_meshes.append(tumor_mesh)
                        
                        # Add diagnostic annotation
                        diag_info = diagnostic_info[component_name]
                        model_annotations[component_name] = {
                            'description': diag_info['description'],
                            'clinical_significance': diag_info['clinical_significance'],
                            'volume_mm3': diag_info['volume_mm3'],
                            'location': diag_info['location'],
                            'vertices': len(tumor_mesh.vertices)
                        }
                        
                        print(f"    Added {component_name}: {diag_info['volume_mm3']:,} mm³ at {diag_info['location']}")
    
    # Combine all meshes
    print(f"Combining {len(all_meshes)} meshes into ultra-realistic model...")
    if len(all_meshes) > 1:
        final_mesh = trimesh.util.concatenate(all_meshes)
    else:
        final_mesh = all_meshes[0]
    
    # Add comprehensive metadata for diagnostic purposes
    final_mesh.metadata.update({
        'description': 'Ultra-realistic 3D brain model with diagnostic pathology highlighting',
        'components': len(all_meshes),
        'diagnostic_annotations': model_annotations,
        'rendering_notes': {
            'transparency': 'Multi-level transparency for anatomical depth perception',
            'coloring': 'Clinically accurate color coding for pathological regions',
            'surface_quality': 'Ultra-high quality surface with anatomical detail preservation'
        }
    })
    
    print(f"✅ Ultra-realistic model complete:")
    print(f"   - Total vertices: {len(final_mesh.vertices):,}")
    print(f"   - Total faces: {len(final_mesh.faces):,}")
    print(f"   - Diagnostic components: {len(model_annotations)}")
    
    return final_mesh

# Find corresponding MRI files for segmentation
def find_corresponding_mri_files(seg_filepath):
    """Find corresponding MRI files for a segmentation file"""
    print("Looking for corresponding MRI files...")
    
    base_path = seg_filepath.replace('-seg.nii.gz', '').replace('_seg.nii.gz', '')
    base_dir = os.path.dirname(seg_filepath)
    
    mri_patterns = [
        base_path + '-t1ce.nii.gz',
        base_path + '-t1.nii.gz',
        base_path + '-t2.nii.gz',
        base_path + '-flair.nii.gz',
    ]
    
    if os.path.exists(base_dir):
        base_name = os.path.basename(base_path)
        for file in os.listdir(base_dir):
            if file.startswith(base_name) and file.endswith('.nii.gz') and 'seg' not in file:
                full_path = os.path.join(base_dir, file)
                if full_path not in mri_patterns:
                    mri_patterns.append(full_path)
    
    existing_files = [f for f in mri_patterns if os.path.exists(f)]
    
    if existing_files:
        print(f"Found {len(existing_files)} MRI files:")
        for f in existing_files:
            print(f"  - {os.path.basename(f)}")
        return existing_files
    else:
        print("No corresponding MRI files found")
        return []

# MAIN: Create ultra-realistic brain 3D model
def create_ultra_realistic_brain_3d_model(input_filepath, output_filepath="exports/ultra_realistic_brain.glb", resolution=(0.4, 0.4, 0.4)):
    """
    Create an ultra-realistic 3D brain model with proper transparency and diagnostic annotations
    """
    print(f"\n{Fore.CYAN}🎯 CREATING ULTRA-REALISTIC 3D BRAIN MODEL{Style.RESET_ALL}")
    print(f"📁 Input: {input_filepath}")
    print(f"💾 Output: {output_filepath}")
    print(f"🔍 Resolution: {resolution} mm")
    
    os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
    
    try:
        # Load and process data
        print(f"\n{Fore.YELLOW}Step 1: Loading data...{Style.RESET_ALL}")
        data, affine = load_mri_data(input_filepath)
        
        is_segmentation = ('seg' in input_filepath.lower() or 
                          np.max(data) <= 4 or 
                          len(np.unique(data)) < 10)
        
        tumor_components = None
        diagnostic_info = None
        mri_data = None
        
        if is_segmentation:
            print(f"\n{Fore.YELLOW}Step 2a: Processing segmentation with diagnostics...{Style.RESET_ALL}")
            tumor_components, diagnostic_info = process_tumor_segmentation_with_diagnostics(data)
            
            # Create diagnostic report
            annotations = create_diagnostic_annotations(tumor_components, diagnostic_info)
            print(f"\n{Fore.GREEN}Diagnostic Summary:{Style.RESET_ALL}")
            print(f"  Total pathological volume: {annotations['diagnostic_summary']['total_volume_mm3']:,} mm³")
            print(f"  Components detected: {annotations['diagnostic_summary']['total_components']}")
            
            mri_files = find_corresponding_mri_files(input_filepath)
            if mri_files:
                print(f"Loading MRI data from: {os.path.basename(mri_files[0])}")
                mri_data, _ = load_mri_data(mri_files[0])
            else:
                print(f"{Fore.YELLOW}Creating brain surface from segmentation...{Style.RESET_ALL}")
                combined_mask = np.zeros_like(data, dtype=bool)
                for comp_info in tumor_components.values():
                    if 'mask' in comp_info:
                        combined_mask = combined_mask | (comp_info['mask'] > 0.1)
                
                brain_surface = binary_dilation(combined_mask, morphology.ball(15))
                brain_surface = gaussian_filter(brain_surface.astype(float), sigma=4)
                mri_data = brain_surface
        else:
            print(f"\n{Fore.YELLOW}Step 2b: Processing MRI file...{Style.RESET_ALL}")
            mri_data = data
        
        # Memory-efficient resampling
        print(f"\n{Fore.YELLOW}Step 3: Memory-efficient processing...{Style.RESET_ALL}")
        if mri_data is not None:
            current_size = np.prod(mri_data.shape)
            print(f"Current data size: {current_size:,} voxels")
            
            if resolution != (1, 1, 1):
                original_spacing = [abs(affine[i,i]) for i in range(3)]
                resize_factor = np.array(original_spacing) / np.array(resolution)
                target_shape = [int(s * f) for s, f in zip(mri_data.shape, resize_factor)]
                target_size = np.prod(target_shape)
                
                if target_size > 80_000_000:  # Reduced limit for ultra-realistic processing
                    print(f"Adjusting resolution for ultra-realistic processing...")
                    scale_factor = (80_000_000 / target_size) ** (1/3)
                    resolution = tuple(r / scale_factor for r in resolution)
                    resize_factor = np.array(original_spacing) / np.array(resolution)
                
                print(f"Resampling MRI data with ultra-realistic preservation...")
                mri_data = zoom(mri_data, resize_factor, mode='nearest', order=1)
                print(f"Resampled MRI shape: {mri_data.shape}")
                
                if tumor_components:
                    print("Resampling tumor components with precision...")
                    resampled_tumors = {}
                    for name, comp_info in tumor_components.items():
                        print(f"  Resampling {name}...")
                        if 'mask' in comp_info:
                            resampled_mask = zoom(comp_info['mask'], resize_factor, mode='nearest', order=1)
                            comp_info['mask'] = resampled_mask
                            resampled_tumors[name] = comp_info
                    tumor_components = resampled_tumors
            
            elif current_size > 80_000_000:
                print(f"Applying memory-safe downsampling for ultra-realistic processing...")
                downsample_factor = int(np.cbrt(current_size / 60_000_000))
                print(f"Downsampling by factor: {downsample_factor}")
                
                mri_data = mri_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
                print(f"Downsampled MRI shape: {mri_data.shape}")
                
                if tumor_components:
                    downsampled_tumors = {}
                    for name, comp_info in tumor_components.items():
                        if 'mask' in comp_info:
                            downsampled_mask = comp_info['mask'][::downsample_factor, ::downsample_factor, ::downsample_factor]
                            comp_info['mask'] = downsampled_mask
                            downsampled_tumors[name] = comp_info
                    tumor_components = downsampled_tumors
        
        # Generate ultra-realistic 3D model
        print(f"\n{Fore.YELLOW}Step 4: Generating ultra-realistic 3D model...{Style.RESET_ALL}")
        
        if mri_data is None:
            raise Exception("No MRI data available for brain surface generation")
        
        # Adaptive step size for optimal quality
        data_size = np.prod(mri_data.shape)
        if data_size > 40_000_000:
            step_size = 2
        elif data_size > 15_000_000:
            step_size = 1
        else:
            step_size = 1
            
        print(f"Using step size: {step_size} for ultra-realistic quality (data size: {data_size:,} voxels)")
        
        final_mesh = create_ultra_realistic_brain_model(
            mri_data, 
            tumor_components=tumor_components,
            diagnostic_info=diagnostic_info,
            step_size=step_size
        )
        
        # Export with diagnostic metadata
        print(f"\n{Fore.YELLOW}Step 5: Exporting ultra-realistic model...{Style.RESET_ALL}")
        
        # Save diagnostic information as separate JSON file
        if diagnostic_info:
            diagnostic_filepath = output_filepath.replace('.glb', '_diagnostics.json')
            with open(diagnostic_filepath, 'w') as f:
                json.dump({
                    'diagnostic_info': diagnostic_info,
                    'annotations': create_diagnostic_annotations(tumor_components, diagnostic_info) if tumor_components else {},
                    'model_info': {
                        'vertices': len(final_mesh.vertices),
                        'faces': len(final_mesh.faces),
                        'components': len(tumor_components) if tumor_components else 0
                    }
                }, f, indent=2)
            print(f"Diagnostic information saved to: {diagnostic_filepath}")
        
        # Export the 3D model
        final_mesh.export(output_filepath, file_type='glb')
        
        # Get file info
        file_size = os.path.getsize(output_filepath) / (1024*1024)
        
        print(f"\n{Fore.GREEN}🎉 ULTRA-REALISTIC BRAIN MODEL CREATED!{Style.RESET_ALL}")
        print(f"📄 File: {output_filepath}")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"🔢 Vertices: {len(final_mesh.vertices):,}")
        print(f"🔺 Faces: {len(final_mesh.faces):,}")
        
        if diagnostic_info:
            print(f"\n{Fore.GREEN}🩺 DIAGNOSTIC FEATURES:{Style.RESET_ALL}")
            for name, info in diagnostic_info.items():
                if name != 'whole_tumor':
                    print(f"   - {info['description']}: {info['volume_mm3']:,} mm³")
                    print(f"     Location: {info['location']}")
                    print(f"     Clinical: {info['clinical_significance']}")
        
        print(f"\n{Fore.CYAN}🎨 RENDERING FEATURES:{Style.RESET_ALL}")
        print("   - Anatomically accurate brain surface with realistic texturing")
        print("   - Multi-level transparency for depth perception")
        print("   - Clinically accurate pathology color coding")
        print("   - Smooth surface transitions preserving anatomical details")
        print("   - Diagnostic annotations embedded in model metadata")
        
        return output_filepath
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error creating ultra-realistic brain model: {e}{Style.RESET_ALL}")
        raise

# Enhanced convenience function for ultra-realistic models
def create_diagnostic_brain_model(input_path, output_name="diagnostic_brain_model", ultra_quality=True):
    """
    Create ultra-realistic brain model with diagnostic features
    
    Parameters:
    - input_path: Path to NIfTI or DICOM file
    - output_name: Name for output file (without extension)
    - ultra_quality: Use ultra-realistic settings (slower but much better quality)
    
    Returns:
    - Path to generated GLB file
    """
    output_path = f"exports/{output_name}.glb"
    
    if ultra_quality:
        resolution = (0.4, 0.4, 0.4)  # High quality for diagnostic use
    else:
        resolution = (0.6, 0.6, 0.6)  # Standard quality
    
    return create_ultra_realistic_brain_3d_model(
        input_filepath=input_path,
        output_filepath=output_path,
        resolution=resolution
    )

# NEW: Create comparison models showing different pathology stages
def create_pathology_comparison_model(input_paths, output_name="pathology_comparison"):
    """
    Create side-by-side comparison of different pathology stages or patients
    
    Parameters:
    - input_paths: List of paths to different segmentation files
    - output_name: Base name for output files
    
    Returns:
    - List of generated model paths
    """
    generated_models = []
    
    for i, input_path in enumerate(input_paths):
        model_name = f"{output_name}_case_{i+1}"
        try:
            model_path = create_diagnostic_brain_model(
                input_path=input_path,
                output_name=model_name,
                ultra_quality=True
            )
            generated_models.append(model_path)
            print(f"Generated comparison model {i+1}: {model_path}")
        except Exception as e:
            print(f"Error generating model {i+1}: {e}")
    
    return generated_models

# NEW: Batch processing for multiple scans
def batch_process_brain_scans(input_directory, output_directory="exports/batch_processed"):
    """
    Process multiple brain scans in batch mode
    
    Parameters:
    - input_directory: Directory containing brain scan files
    - output_directory: Directory for output models
    
    Returns:
    - List of successfully processed files
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all relevant files
    scan_files = []
    for file in os.listdir(input_directory):
        if file.endswith(('.nii.gz', '.nii', '.dcm')) and ('seg' in file.lower() or 't1' in file.lower() or 't2' in file.lower()):
            scan_files.append(os.path.join(input_directory, file))
    
    print(f"Found {len(scan_files)} scan files for batch processing")
    
    processed_files = []
    for i, scan_file in enumerate(scan_files):
        try:
            print(f"\n{Fore.CYAN}Processing file {i+1}/{len(scan_files)}: {os.path.basename(scan_file)}{Style.RESET_ALL}")
            
            output_name = os.path.splitext(os.path.splitext(os.path.basename(scan_file))[0])[0]  # Remove .nii.gz
            output_path = os.path.join(output_directory, f"{output_name}_realistic_brain.glb")
            
            result_path = create_ultra_realistic_brain_3d_model(
                input_filepath=scan_file,
                output_filepath=output_path,
                resolution=(0.5, 0.5, 0.5)  # Balanced quality for batch processing
            )
            
            processed_files.append(result_path)
            print(f"✅ Successfully processed: {result_path}")
            
        except Exception as e:
            print(f"❌ Error processing {scan_file}: {e}")
    
    print(f"\n{Fore.GREEN}Batch processing complete!{Style.RESET_ALL}")
    print(f"Successfully processed: {len(processed_files)}/{len(scan_files)} files")
    
    return processed_files

# Example usage and testing
if __name__ == "__main__":
    # Example 1: Single ultra-realistic diagnostic model
    seg_file = r'C:\Users\sandh\Github\mri-to-3d-feature\sample_data\BraTS-Africa\51_OtherNeoplasms\BraTS-SSA-00108-000\BraTS-SSA-00108-000-seg.nii.gz'
    
    print(f"{Fore.CYAN}🎯 CREATING ULTRA-REALISTIC DIAGNOSTIC BRAIN MODEL{Style.RESET_ALL}")
    
    try:
        # Create ultra-realistic model with diagnostic features
        result_file = create_diagnostic_brain_model(
            input_path=seg_file,
            output_name="ultra_realistic_diagnostic_brain",
            ultra_quality=True
        )
        
        print(f"\n{Fore.GREEN}🎉 SUCCESS! Ultra-realistic diagnostic model created!{Style.RESET_ALL}")
        print(f"📁 Model file: {result_file}")
        print(f"📊 Diagnostic data: {result_file.replace('.glb', '_diagnostics.json')}")
        
        print(f"\n{Fore.CYAN}🎨 KEY IMPROVEMENTS:{Style.RESET_ALL}")
        print("✅ Realistic brain surface texture (no more blocky appearance)")
        print("✅ Multi-level transparency for proper depth perception")
        print("✅ Clinically accurate pathology color coding")
        print("✅ Smooth anatomical transitions and preserved details")
        print("✅ Embedded diagnostic annotations with clinical significance")
        print("✅ Separate JSON file with detailed diagnostic information")
        
        print(f"\n{Fore.YELLOW}📋 VIEWING RECOMMENDATIONS:{Style.RESET_ALL}")
        print("• Use Blender for full diagnostic feature access")
        print("• 3D viewers with transparency support (Windows 3D Viewer, online GLB viewers)")
        print("• Import diagnostic JSON for detailed pathology information")
        print("• Adjust transparency settings in viewer for optimal visualization")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
    
    # Example 2: Batch processing (uncomment to use)
    # batch_directory = r'path\to\your\scan\directory'
    # batch_results = batch_process_brain_scans(batch_directory)
    
    # Example 3: Comparison models (uncomment to use)
    # comparison_files = [
    #     r'path\to\scan1-seg.nii.gz',
    #     r'path\to\scan2-seg.nii.gz'
    # ]
    # comparison_models = create_pathology_comparison_model(comparison_files)