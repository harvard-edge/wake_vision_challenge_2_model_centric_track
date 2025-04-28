# 🚀 **Model-Centric Track**

Welcome to the **Model-Centric Track** of the **Wake Vision Challenge 2**! 🎉

This track challenges you to **push the boundaries of tiny computer vision** by designing innovative model architectures for the newly released [Wake Vision Dataset](https://wakevision.ai/).

🔗 **Learn More**: Wake Vision Challenge Details (TBD)

---

## 🌟 **Challenge Overview**

Participants are invited to:

1. **Design novel model architectures** to achieve high accuracy.
2. Optimize for **resource efficiency** (e.g., memory, inference time).
3. Evaluate models on the **public test set** of the Wake Vision dataset.

You can modify the **model architecture** freely, but the **dataset must remain unchanged**. 🛠️

---

## 🛠️ **Getting Started**

### Step 1: Install Docker Engine 🐋

First, install Docker on your machine:
- [Install Docker Engine](https://docs.docker.com/engine/install/).

---

### Step 2: Download the Wake Vision dataset

1. [Sign up](https://dataverse.harvard.edu/dataverseuser.xhtml;jsessionid=b78ff6ae13347e089bc776b916e9?editMode=CREATE&redirectPage=%2Fdataverse_homepage.xhtml) on Harvard Dataverse

2. On your account information page go to the API Token tab and create a new API Token for Harvard Dataverse

3. Substitute "your-api-token-goes-here" with your API token in the following command and run it inside the directory where you cloned this repository to download and build the Wake Vision Dataset:

```bash
sudo docker run -it --rm -v "$(pwd):/tmp" -w /tmp tensorflow/tensorflow:2.19.0 python download_and_build_wake_vision_dataset.py your-api-token-goes-here
```

💡 **Note**: Make sure to have at least 600 GB of free disk space.

---

### 💻 **Running Without a GPU**

Run the following command inside the directory where you cloned this repository:

```bash
sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:2.19.0 python model_centric.py
```

- This trains the [ColabNAS model](https://github.com/harvard-edge/Wake_Vision/blob/main/experiments/comprehensive_model_architecture_experiments/wake_vision_quality/k_8_c_5.py), a state-of-the-art person detection model, on the Wake Vision dataset.
- Modify the `model_centric.py` script to propose your own architecture.

---

### ⚡ **Running With a GPU**

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Verify your [GPU drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation).

Run the following command inside the directory where you cloned this repository:

```bash
sudo docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:2.19.0-gpu python model_centric.py
```

- This trains the [ColabNAS model](https://github.com/harvard-edge/Wake_Vision/blob/main/experiments/comprehensive_model_architecture_experiments/wake_vision_quality/k_8_c_5.py) on the Wake Vision dataset.
- Modify the `model_centric.py` script to design your own model architecture.

💡 **Note**: The first execution may take several hours as it downloads the full dataset (~365 GB).

---

## 🎯 **Tips for Success**

- **Focus on Model Innovation**: Experiment with architecture design, layer configurations, and optimization techniques.
- **Stay Efficient**: Resource usage is critical—consider model size, inference time, and memory usage.
- **Collaborate**: Join the community discussions on Discord (TBD) to exchange ideas and insights!

---

## 📚 **Resources**

- [ColabNAS Model Documentation](https://github.com/AndreaMattiaGaravagno/ColabNAS)
- [Docker Documentation](https://docs.docker.com/)
- [Wake Vision Dataset](https://wakevision.ai/)

---

## 📞 **Contact Us**

Have questions or need help? Reach out on [Discord](https://discord.com/channels/803180012572114964/1323721491736432640).

---

🌟 **Happy Innovating and Good Luck!** 🌟
