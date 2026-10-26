"""Download the official 224-pixel MedMNIST data used by the examples."""
import argparse
import json
from pathlib import Path
import medmnist
import lern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/medmnist')
    parser.add_argument('--tasks', default='all')
    args = parser.parse_args()
    tasks = json.loads((Path(lern.__file__).parent/'medmnist_classes.json').read_text())
    requested = list(tasks) if args.tasks=='all' else args.tasks.split(',')
    Path(args.root).mkdir(parents=True, exist_ok=True)
    for task in requested:
        if task not in tasks:
            raise ValueError(f'Unsupported task: {task}')
        dataset = getattr(medmnist, medmnist.INFO[task]['python_class'])
        dataset(root=args.root, split='test', size=224, download=True)
        print('Ready:', task)

if __name__ == '__main__':
    main()
