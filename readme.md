<h1>ML Learning Project - Unsupervised</h1>
<p>
    This project focuses on understanding and applying unsupervised learning algorithm — primarily K-Means Clustering and Principal Component Analysis (PCA) — to practical usecase such as image color reduction and dimensionality reduction.
</p>

### Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Modules used](#modules-used)
- [References](#references)

## Installation

1. Clone the repo

```bash
git clone git@github.com:mohitdixit02/ml_learning_project_unsupervised.git
```

2. Setup a python (>=3.13) virtual environment (Snippet for Windows)

```bash
cd ml_learning_project_unsupervised
python -m venv venv
```

3. Activate the environment and install the dependencies

```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the main script

```bash
python main.py
```

## Project Overview
<p>
    <ul>
        <li>The project consists of two primary components:
            <ol>
                <li><strong>Image Color Reduction:</strong> Utilizes K-Means Clustering to group similar pixels and reduce the image to a fixed number of dominant colors. An optional PCA (Principal Component Analysis) step is also included to reduce feature dimensions before clustering.</li>
                <li><strong>Text Enhancement:</strong> The image is divided into smaller pixel windows, and K-Means Clustering is applied within each window to identify the two most dominent colors. If the color difference exceeds 2%, the window's pixels are recolored to improve consistency and enhance clarity.</li>
            </ol>
        </li>
        <li>After processing, a preview of the output image is displayed using <code>pyplot</code>. The final processed image is saved in the root directory upon program completion.</li>
        <li>Sample input images are available in the <code>input</code> folder for testing purposes.</li>
    </ul>
</p>

## Modules Used
<ol>
    <li>Numpy</li>
    <li>Matplotlib</li>
    <li>OpenCV</li>
    <li>Scikit-learn</li>
</ol>

## References
<p>Sample Images in the input folder are taken from following sources:</p>
<ol>
    <li><strong>"bird.jpg" - </strong>used in image color reduction - <a href="https://unsplash.com/photos/a-couple-of-colorful-birds-Mkv2aKWHx00?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash">source</a></li>
    <li><strong>"sign.jpg" - </strong>used in text enhancement - <a href="https://elements-resized.envatousercontent.com/elements-cover-images/da9b30a5-7823-49ee-9440-1fa73c689ae7?w=2038&cf_fit=scale-down&q=85&format=auto&s=c663a4d16707d1c1e73707539a5d3227f42cca8ce3ca9d934a09300c070c2197">source</a></li>
</ol>

<h4>- Thanks -</h4>
For any query, email at: mohit.vsht@gmail.com - Mohit Sharma