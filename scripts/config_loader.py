import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build derived paths
    outs = config['paths']['spaceranger_outs']
    micro = config['paths']['microscopy']

    sample_folders = [s['id'] for s in config['samples']]
    sample_paths = [f"{outs}/{s}/outs" for s in sample_folders]
    id_to_image = {s['id']: f"{micro}/{s['image']}" for s in config['samples']}
    sample_mapping = {s['id']: s['name'] for s in config['samples']}
    
    return {
        'sample_folders': sample_folders,
        'sample_paths': sample_paths,
        'id_to_image': id_to_image,
        'sample_mapping': sample_mapping,
        'paths': config['paths'],
        'params': config.get('params', {}),
        'marker_genes': config.get('marker_genes', {}),
        'marker_genes_grouped': config.get('marker_genes_grouped', {})
    }



