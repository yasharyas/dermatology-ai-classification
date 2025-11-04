"""
Data Preparation Script for HAM10000 Dataset
- Analyzes class distribution
- Creates train/val splits
- Generates metadata files
- Computes dataset statistics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')

class HAM10000DataPreparation:
    def __init__(self, root_dir='d:/CODING/dataverse_files'):
        self.root_dir = Path(root_dir)
        self.metadata_file = self.root_dir / 'HAM10000_metadata'
        self.output_dir = self.root_dir / 'processed'
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mapping
        self.class_names = {
            'mel': 'Melanoma',
            'nv': 'Melanocytic nevi',
            'bkl': 'Benign keratosis',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        
        self.class_to_idx = {k: i for i, k in enumerate(self.class_names.keys())}
        
    def load_metadata(self):
        """Load and process metadata"""
        print("Loading metadata...")
        df = pd.read_csv(self.metadata_file)
        print(f"Total samples: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        return df
    
    def analyze_class_distribution(self, df):
        """Analyze and visualize class distribution"""
        print("\n" + "="*80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*80)
        
        class_counts = df['dx'].value_counts()
        print("\nClass counts:")
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cls:6s} ({self.class_names[cls]:25s}): {count:5d} ({percentage:5.2f}%)")
        
        # Calculate imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
        print(f"⚠️  Class imbalance detected! Will use balanced sampling strategies.")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        ax = axes[0]
        bars = ax.bar(range(len(class_counts)), class_counts.values, color='steelblue')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels([self.class_names[c] for c in class_counts.index], rotation=45, ha='right')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution (Imbalanced)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        # Pie chart
        ax = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        wedges, texts, autotexts = ax.pie(class_counts.values, 
                                           labels=[self.class_names[c] for c in class_counts.index],
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90)
        ax.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {self.output_dir / 'class_distribution.png'}")
        plt.close()
        
        return class_counts
    
    def analyze_metadata_features(self, df):
        """Analyze age, sex, localization distributions"""
        print("\n" + "="*80)
        print("METADATA FEATURES ANALYSIS")
        print("="*80)
        
        # Age analysis
        print("\nAge statistics:")
        print(df['age'].describe())
        print(f"Missing age values: {df['age'].isna().sum()}")
        
        # Sex distribution
        print("\nSex distribution:")
        print(df['sex'].value_counts())
        
        # Localization distribution
        print("\nLocalization distribution:")
        loc_counts = df['localization'].value_counts()
        for loc, count in loc_counts.items():
            print(f"  {loc:20s}: {count:5d}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age histogram
        ax = axes[0, 0]
        df['age'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution')
        ax.axvline(df['age'].median(), color='red', linestyle='--', label=f'Median: {df["age"].median():.1f}')
        ax.legend()
        
        # Sex distribution
        ax = axes[0, 1]
        sex_counts = df['sex'].value_counts()
        ax.bar(sex_counts.index, sex_counts.values, color=['lightcoral', 'lightblue', 'lightgray'])
        ax.set_ylabel('Count')
        ax.set_title('Sex Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # Localization distribution
        ax = axes[1, 0]
        loc_counts.plot(kind='barh', ax=ax, color='mediumseagreen')
        ax.set_xlabel('Count')
        ax.set_title('Anatomical Localization Distribution')
        ax.grid(axis='x', alpha=0.3)
        
        # Age by diagnosis
        ax = axes[1, 1]
        df.boxplot(column='age', by='dx', ax=ax)
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Age')
        ax.set_title('Age Distribution by Diagnosis')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metadata_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {self.output_dir / 'metadata_analysis.png'}")
        plt.close()
    
    def find_all_images(self):
        """Find all image files and create mapping"""
        print("\n" + "="*80)
        print("FINDING ALL IMAGES")
        print("="*80)
        
        image_dirs = [
            self.root_dir / 'HAM10000_images_part_1',
            self.root_dir / 'HAM10000_images_part_2'
        ]
        
        all_images = {}
        for img_dir in image_dirs:
            if img_dir.exists():
                images = list(img_dir.glob('*.jpg'))
                print(f"Found {len(images)} images in {img_dir.name}")
                for img_path in images:
                    image_id = img_path.stem  # ISIC_XXXXXXX
                    all_images[image_id] = str(img_path)
        
        print(f"\nTotal unique images found: {len(all_images)}")
        return all_images
    
    def create_train_val_split(self, df, image_mapping, val_split=0.15, random_state=42):
        """Create stratified train/val split"""
        print("\n" + "="*80)
        print("CREATING TRAIN/VAL SPLIT")
        print("="*80)
        
        # Filter to only images that exist
        df_filtered = df[df['image_id'].isin(image_mapping.keys())].copy()
        print(f"Samples with existing images: {len(df_filtered)}")
        
        # Add image paths
        df_filtered['image_path'] = df_filtered['image_id'].map(image_mapping)
        
        # Add numeric labels
        df_filtered['label'] = df_filtered['dx'].map(self.class_to_idx)
        
        # Stratified split
        train_df, val_df = train_test_split(
            df_filtered,
            test_size=val_split,
            stratify=df_filtered['dx'],
            random_state=random_state
        )
        
        print(f"\nTrain set: {len(train_df)} samples")
        print(f"Val set: {len(val_df)} samples")
        
        print("\nTrain class distribution:")
        print(train_df['dx'].value_counts())
        
        print("\nVal class distribution:")
        print(val_df['dx'].value_counts())
        
        # Save splits
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        
        print(f"\n✅ Saved train split: {self.output_dir / 'train.csv'}")
        print(f"✅ Saved val split: {self.output_dir / 'val.csv'}")
        
        return train_df, val_df
    
    def compute_class_weights(self, df):
        """Compute class weights for balanced training"""
        class_counts = df['dx'].value_counts()
        total = len(df)
        
        # Inverse frequency weighting
        weights = {cls: total / (len(class_counts) * count) 
                  for cls, count in class_counts.items()}
        
        # Convert to list ordered by class_to_idx
        weight_list = [weights[cls] for cls in self.class_names.keys()]
        
        return weight_list
    
    def save_statistics(self, df, train_df, val_df, class_counts):
        """Save dataset statistics"""
        stats = {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'class_distribution': {k: int(v) for k, v in class_counts.items()},
            'train_class_weights': self.compute_class_weights(train_df),
            'age_range': [float(df['age'].min()), float(df['age'].max())],
            'age_mean': float(df['age'].mean()),
            'age_std': float(df['age'].std()),
            'sex_distribution': df['sex'].value_counts().to_dict(),
            'localization_distribution': df['localization'].value_counts().to_dict()
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Saved statistics: {self.output_dir / 'dataset_stats.json'}")
        return stats
    
    def run(self):
        """Run complete data preparation pipeline"""
        print("\n" + "="*80)
        print("HAM10000 DATA PREPARATION PIPELINE")
        print("="*80)
        
        # Load metadata
        df = self.load_metadata()
        
        # Analyze class distribution
        class_counts = self.analyze_class_distribution(df)
        
        # Analyze metadata features
        self.analyze_metadata_features(df)
        
        # Find all images
        image_mapping = self.find_all_images()
        
        # Create train/val split
        train_df, val_df = self.create_train_val_split(df, image_mapping)
        
        # Save statistics
        stats = self.save_statistics(df, train_df, val_df, class_counts)
        
        print("\n" + "="*80)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - train.csv ({len(train_df)} samples)")
        print(f"  - val.csv ({len(val_df)} samples)")
        print(f"  - dataset_stats.json")
        print(f"  - class_distribution.png")
        print(f"  - metadata_analysis.png")
        
        print("\n📊 Key Statistics:")
        print(f"  - Total samples: {stats['total_samples']}")
        print(f"  - Train/Val split: {len(train_df)}/{len(val_df)}")
        print(f"  - Most common class: {class_counts.index[0]} ({class_counts.values[0]} samples)")
        print(f"  - Least common class: {class_counts.index[-1]} ({class_counts.values[-1]} samples)")
        print(f"  - Imbalance ratio: {class_counts.values[0] / class_counts.values[-1]:.2f}x")
        
        print("\n🎯 Next Steps:")
        print("  1. Review class_distribution.png to understand data imbalance")
        print("  2. Review metadata_analysis.png for feature distributions")
        print("  3. Run EDA notebook: notebooks/01_eda.ipynb")
        print("  4. Start training: python src/train.py")
        
        return stats


if __name__ == '__main__':
    preparer = HAM10000DataPreparation()
    stats = preparer.run()
