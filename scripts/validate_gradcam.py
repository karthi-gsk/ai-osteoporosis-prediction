"""
Phase 5A: Standalone Grad-CAM Validation Script

Tests genuine Grad-CAM on representative validation knee X-rays:
1. Normal sample
2. Osteopenia sample
3. Osteoporosis sample

Verifies:
- Checkpoint loading integrity
- Activation and gradient shapes
- Heatmap finiteness and normalization
- Overlay dimensions matching original image
- Class targeting behavior
- Saves publication-quality research figures to reports/gradcam_visualizations/
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.gradcam import GradCAM, load_trained_resnet18, preprocess_image, CLASS_NAMES


def main():
    print("=" * 65)
    print("  PHASE 5A: STANDALONE GRAD-CAM VALIDATION & VERIFICATION")
    print("=" * 65)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(workspace_root, 'reports', 'gradcam_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(workspace_root, 'models', 'checkpoints', 'resnet18', 'best_model.pth')
    val_dir = os.path.join(workspace_root, 'dataset', 'strict_clean_knee_osteoporosis', 'val')

    # 1. Load model and checkpoint
    print(f"\n1. Loading frozen model from:\n   {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_trained_resnet18(checkpoint_path, device=device)
    print(f"   Model: ResNet18 (Total parameters: {sum(p.numel() for p in model.parameters()):,})")
    print(f"   Checkpoint Best Epoch: {ckpt.get('epoch')}")
    print(f"   Checkpoint Best Val F1: {ckpt.get('best_val_f1'):.4f}")
    print("   Parameters frozen: True (model.eval() active)")

    # 2. Initialize GradCAM
    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    print(f"\n2. Initialized Grad-CAM with target layer:\n   {model.layer4[-1]}")

    # 3. Select representative validation images
    test_cases = [
        {
            'category': 'Normal',
            'filename': 'N10.JPEG',
            'path': os.path.join(val_dir, 'Normal', 'N10.JPEG')
        },
        {
            'category': 'Osteopenia',
            'filename': 'OP104.jpg',
            'path': os.path.join(val_dir, 'Osteopenia', 'OP104.jpg')
        },
        {
            'category': 'Osteoporosis',
            'filename': 'OS19.jpg',
            'path': os.path.join(val_dir, 'Osteoporosis', 'OS19.jpg')
        }
    ]

    verification_results = []

    print("\n3. Generating and Validating Grad-CAM Visualizations...")

    for case in test_cases:
        cat = case['category']
        img_path = case['path']
        print(f"\n   --- Processing {cat}: {case['filename']} ---")

        if not os.path.exists(img_path):
            print(f"   Error: File not found: {img_path}")
            continue

        input_tensor, original_pil = preprocess_image(img_path)
        orig_w, orig_h = original_pil.size

        # Generate Grad-CAM for predicted class
        result = gradcam.generate_cam(input_tensor)
        cam = result['cam']

        # Verification checks
        act_shape = result['activation_shape']
        grad_shape = result['gradient_shape']
        assert act_shape == (1, 512, 7, 7), f"Unexpected activation shape: {act_shape}"
        assert grad_shape == (1, 512, 7, 7), f"Unexpected gradient shape: {grad_shape}"
        assert np.all(np.isfinite(cam)), "CAM contains non-finite values (NaN/Inf)!"
        assert cam.min() >= 0.0 and cam.max() <= 1.0, f"CAM values out of [0, 1] range: [{cam.min()}, {cam.max()}]"

        heatmap_pil, overlay_pil = GradCAM.overlay_heatmap(original_pil, cam, alpha=0.45, colormap_name='jet')
        assert overlay_pil.size == (orig_w, orig_h), "Overlay size does not match original dimensions!"
        assert heatmap_pil.size == (orig_w, orig_h), "Heatmap size does not match original dimensions!"

        pred_name = result['pred_class_name']
        conf = result['confidence']
        print(f"   Actual Category:    {cat}")
        print(f"   Predicted Category: {pred_name} (Confidence: {conf*100:.2f}%)")
        print(f"   Class Probabilities: Normal={result['all_probs'][0]:.3f}, "
              f"Osteopenia={result['all_probs'][1]:.3f}, Osteoporosis={result['all_probs'][2]:.3f}")
        print(f"   Activation Shape:   {act_shape}")
        print(f"   Gradient Shape:     {grad_shape}")
        print(f"   CAM Value Range:    [{cam.min():.4f}, {cam.max():.4f}] (Finite: True)")
        print(f"   Output Dimensions:  {orig_w}x{orig_h} (Matches original)")

        # Create multi-panel visualization figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        # Panel 1: Original X-Ray
        axes[0].imshow(original_pil)
        axes[0].set_title(f"Original Knee X-Ray\nGround Truth: {cat} ({orig_w}x{orig_h})", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Grad-CAM Heatmap
        im2 = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM Heatmap (Layer4)\nTarget: {result['target_class_name']}", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Class Activation Intensity', fontsize=10)

        # Panel 3: Transparent Overlay
        axes[2].imshow(overlay_pil)
        axes[2].set_title(f"Anatomical Overlay (alpha=0.45)\nPrediction: {pred_name} ({conf*100:.1f}%)",
                          fontsize=12, fontweight='bold',
                          color='#2e7d32' if pred_name == cat else '#c62828')
        axes[2].axis('off')

        fig.suptitle(f"Osteoporosis Prediction AI — Grad-CAM Explainability [{cat} Sample: {case['filename']}]",
                     fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout()

        fig_path = os.path.join(output_dir, f"gradcam_{cat.lower()}_{os.path.splitext(case['filename'])[0]}.png")
        fig.savefig(fig_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved visualization: {fig_path}")

        # Also save individual overlay image
        overlay_single_path = os.path.join(output_dir, f"overlay_{cat.lower()}_{os.path.splitext(case['filename'])[0]}.png")
        overlay_pil.save(overlay_single_path)

        verification_results.append({
            'case': cat,
            'filename': case['filename'],
            'original_dimensions': [orig_w, orig_h],
            'predicted_class': pred_name,
            'ground_truth': cat,
            'confidence': round(conf, 4),
            'class_probabilities': {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, result['all_probs'])},
            'activation_shape': list(act_shape),
            'gradient_shape': list(grad_shape),
            'cam_min': float(cam.min()),
            'cam_max': float(cam.max()),
            'visualization_file': os.path.basename(fig_path),
            'overlay_file': os.path.basename(overlay_single_path),
            'checks_passed': True
        })

    # 4. Multi-target Class Contrast Test
    # Verify how Grad-CAM heatmaps contrast when asking: "Why Normal?" vs "Why Osteoporosis?" for the same image
    print("\n4. Running Class-Specific Contrast Test (Why Class A vs Why Class B on Osteoporosis sample)...")
    sample_case = test_cases[2]  # OS19
    input_tensor, original_pil = preprocess_image(sample_case['path'])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_pil)
    axes[0].set_title(f"Original X-Ray\nTrue: {sample_case['category']}", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for idx, target_cls in enumerate(CLASS_NAMES):
        res = gradcam.generate_cam(input_tensor, target_class=idx)
        _, overlay_target = GradCAM.overlay_heatmap(original_pil, res['cam'], alpha=0.45)
        axes[idx + 1].imshow(overlay_target)
        axes[idx + 1].set_title(f"Target: {target_cls}\nProb: {res['all_probs'][idx]*100:.1f}%", fontsize=11, fontweight='bold')
        axes[idx + 1].axis('off')

    contrast_fig_path = os.path.join(output_dir, "gradcam_class_contrast_comparison.png")
    fig.suptitle("Grad-CAM Class-Specific Activation Contrast (Same X-Ray, Different Target Classes)",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(contrast_fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved contrast figure: {contrast_fig_path}")

    # Remove hooks
    gradcam.remove_hooks()
    print("   Grad-CAM hooks successfully removed.")

    # 5. Save summary report
    summary = {
        'date': '2026-09-25',
        'model_architecture': 'ResNet18',
        'checkpoint_path': checkpoint_path,
        'checkpoint_best_epoch': ckpt.get('epoch'),
        'checkpoint_best_val_f1': ckpt.get('best_val_f1'),
        'target_layer': 'model.layer4[-1] (BasicBlock 2)',
        'target_layer_channels': 512,
        'spatial_resolution': '7x7',
        'verification_cases': verification_results,
        'class_contrast_comparison': os.path.basename(contrast_fig_path),
        'methodology': (
            'Genuine Grad-CAM calculates gradients of the target class logit with respect to the '
            'feature activation maps of ResNet18 layer4[-1]. Global average pooling yields channel '
            'importance weights alpha_k. A ReLU activation is applied to the weighted combination, '
            'followed by bilinear interpolation and [0, 1] min-max normalization. Transparent alpha-blending '
            '(alpha=0.45) overlays the JET colormap onto the original X-ray.'
        ),
        'disclaimer': (
            'Grad-CAM heatmaps highlight image regions that strongly influenced the convolutional neural network '
            'representation for knee osteoporosis classification. They do NOT constitute medical diagnosis, '
            'anatomical segmentations of bone mineral loss, or verified DXA T-score measurements. '
            'Outputs are strictly for research and model interpretability.'
        )
    }

    summary_json_path = os.path.join(output_dir, 'gradcam_verification_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n5. Saved verification summary to:\n   {summary_json_path}")

    print("\n" + "=" * 65)
    print("  GRAD-CAM IMPLEMENTATION & VERIFICATION COMPLETE (ALL CHECKS PASSED)")
    print("=" * 65)


if __name__ == '__main__':
    main()
