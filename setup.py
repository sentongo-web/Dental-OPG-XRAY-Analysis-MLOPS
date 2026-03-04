from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dental-opg-cavity-detection",
    version="1.0.0",
    author="Paul Sentongo",
    author_email="paul.sentongo@example.com",
    description="MLOps pipeline for dental cavity detection in OPG X-ray images using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulsentongo/Dental-OPG-XRAY-Analysis-MLOPS",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "dental-train=dental_opg.pipeline.stage_04_model_training:main",
            "dental-predict=dental_opg.pipeline.prediction_pipeline:main",
        ],
    },
)
