from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def evaluate(original, filtered):
    return (
        psnr(original, filtered, data_range=1.0),
        ssim(original, filtered, data_range=1.0)
    )