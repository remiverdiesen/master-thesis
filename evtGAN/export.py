import torch
import pandas as pd
from models.evtgan import Generator
from utils.data_utils import get_relevant_points

def export_synthetic_data(config_file, checkpoint_path, n_samples=10000):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')
    _, ids = load_and_preprocess_data(config['data']['data_file'], config['data']['ids_file'])
    ids = ids.to(device)

    generator = Generator(config['model']['noise_dim']).to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    synthetic_data = []
    with torch.no_grad():
        for _ in range(0, n_samples, config['model']['batch_size']):
            batch_size = min(config['model']['batch_size'], n_samples - len(synthetic_data))
            z = torch.randn(batch_size, config['model']['noise_dim']).to(device)
            fake_data = generator(z)
            points = get_relevant_points(fake_data, ids)
            synthetic_data.append(points.cpu().numpy())
    synthetic_data = np.concatenate(synthetic_data, axis=0)[:n_samples]

    df = pd.DataFrame(synthetic_data)
    output_path = f"{config['data']['output_dir']}/synthetic_precipitation.csv"
    df.to_csv(output_path, header=None, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    export_synthetic_data("config/config.yaml", "data/synthetic/generator_epoch_29999.pt")