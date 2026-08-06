"""Energy OOD AUROC for an ImageFolder ID set versus standard image OOD sets."""
import argparse, json, os
from copy import deepcopy

import torch
from diffusers.models import AutoencoderKL
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from download import find_model
from models import EqM_models


class Constant(Dataset):
    def __init__(self, n, size): self.n, self.size = n, size
    def __len__(self): return self.n
    def __getitem__(self, i):
        v = torch.tensor(2 * (i + .5) / self.n - 1)
        return v.expand(3, self.size, self.size), 0


def load(args, device):
    model = EqM_models[args.model](input_size=args.image_size // 8,
        num_classes=args.num_classes, uncond=True, ebm=args.ebm).to(device)
    ema = deepcopy(model).to(device)
    state = find_model(args.ckpt)
    ema.load_state_dict(state.get("ema", state))
    ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    return ema, vae


def data(name, root, transform, n, size):
    if name == "cifar10": return datasets.CIFAR10(root, train=False, download=True, transform=transform)
    if name == "svhn": return datasets.SVHN(root, split="test", download=True, transform=transform)
    if name == "dtd": return datasets.DTD(root, split="test", download=True, transform=transform)
    if name == "constant": return Constant(n, size)
    raise ValueError(name)


def energies(dataset, model, vae, args, device):
    if args.num_images:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_images]
        dataset = Subset(dataset, indices.tolist())
    values = []
    for x, _ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers):
        x = x.to(device)
        with torch.no_grad(): x = vae.encode(x).latent_dist.mean.mul_(0.18215)
        y = torch.full((len(x),), args.num_classes, device=device, dtype=torch.long)
        t = torch.zeros(len(x), device=device)
        with torch.enable_grad(): _, e = model(x, t, y, get_energy=True)
        values.append(e.detach().cpu())
        if args.num_images and sum(map(len, values)) >= args.num_images: break
    return torch.cat(values)[:args.num_images or None]


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    model, vae = load(args, device)
    id_energy = energies(datasets.ImageFolder(args.id_root, transform), model, vae, args, device)
    result = {"checkpoint": args.ckpt, "ebm": args.ebm, "id": args.id_root, "num_id": len(id_energy), "auroc": {}}
    for name in args.ood:
        ood_energy = energies(data(name, args.data_root, transform, len(id_energy), args.image_size), model, vae, args, device)
        # Lower energy is ID, so negate it for sklearn's positive=OOD convention.
        labels = torch.cat([torch.zeros(len(id_energy)), torch.ones(len(ood_energy))])
        scores = -torch.cat([id_energy, ood_energy])
        result["auroc"][name] = roc_auc_score(labels, scores)
    result["auroc"]["avg"] = sum(result["auroc"].values()) / len(result["auroc"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--id-root", required=True, help="ImageFolder-format ImageNet validation set")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="eval_results/ood_energy.json")
    p.add_argument("--model", choices=EqM_models.keys(), default="EqM-B/4")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--vae", choices=["ema", "mse"], default="ema")
    p.add_argument("--ebm", choices=["dot", "direct", "forward-backwards-direct"], default="dot")
    p.add_argument("--ood", nargs="+", choices=["cifar10", "svhn", "dtd", "constant"], default=["cifar10", "svhn", "dtd", "constant"])
    p.add_argument("--num-images", type=int, default=0, help="0 uses each dataset's full test split")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
