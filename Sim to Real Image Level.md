## Overview
Transforming satellite imagery into map representations is a crucial task in various domains, including urban planning, environmental monitoring, and geographic information systems (GIS). Generative Adversarial Networks (GANs), particularly Pix2Pix and CycleGAN, have proven effective for this image-to-image translation task.

---

### 1.Pix2Pix: Conditional GAN for Paired Image Translation

- **Advantages** :
    1. Paired Data Requirement: Requires datasets with corresponding satellite and map images.
    2. High-Resolution Outputs: Produces detailed and accurate map representations.
    3. Stable Training: Utilizes U-Net architecture with PatchGAN discriminator for stable training.

- **Implementation**
Libraries: PyTorch, torchvision, matplotlib.

- (https://medium.com/%40pbvaras/from-satellite-to-maps-using-deep-learning-a-step-by-step-guide-b22fe5f10ac9)

- (https://www.researchgate.net/figure/Given-satellite-imagery-data-x-and-corresponding-map-data-y-illustration-of-our-CycleGAN_fig4_341538965)

- (https://www.codegenes.net/blog/pix2pix-pytorch-github/)

- (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

- (https://github.com/anh-nn01/Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework)


- **Architecture** :
- Generator: U-Net with skip connections to preserve spatial information.

- Discriminator: PatchGAN that classifies each patch of the image as real or fake.

---

### 2.CycleGAN: Unpaired Image Translation

- Unpaired Data Flexibility: Does not require paired datasets, making it suitable for diverse image domains.

- Cycle Consistency: Ensures that translating an image to another domain and back yields the original image


- **Architecture** :
- Generators: Two networks that translate images between two domains.

- Discriminators: Each evaluates the authenticity of images in its respective domain.

- Libraries: TensorFlow, Keras.

- (https://junyanz.github.io/CycleGAN/)

- (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

