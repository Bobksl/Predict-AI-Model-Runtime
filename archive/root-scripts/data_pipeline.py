from enhanced_loader import TpugraphsDataset, create_dataloaders
from data_cleaning import DataCleaner, CleaningConfig
from data_exploration import explore_dataset
import pandas as pd

# Step 1: Explore the data
print("Exploring dataset...")
df = explore_dataset("./data/tpugraphs")
print(f"Found {len(df)} files")

# Step 2: Clean the data
print("\nCleaning data...")
config = CleaningConfig()
cleaner = DataCleaner(config)
report = cleaner.clean_directory(
    "./data/tpugraphs",
    output_dir="./data/cleaned"
)
print(f"Cleaned {report.files_modified} files")

# Step 3: Create dataloaders
print("\nCreating dataloaders...")
train_loader, valid_loader = create_dataloaders(
    "./data/cleaned",
    collection="tile:xla",
    batch_size=16,
    num_workers=4,
)

# Step 4: Iterate through data
print("\nSample iteration:")
for i, batch in enumerate(train_loader):
    if i >= 2:
        break
    print(f"Batch {i}: node_feat shape = {batch['node_feat'].shape}")