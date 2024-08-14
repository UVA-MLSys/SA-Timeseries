FROM python:3.10
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get -y update
RUN python -m pip install pip>=24.*

RUN pip install matplotlib numpy pandas scikit-learn sympy tqdm scipy einops
RUN pip install time-interpret captum
RUN pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install pytorch-lightning==1.8.6 \\
        ipykernel \
        ipywidgets \
        jupyter_client \
        notebook \
        numpy \
        setuptools \
        reformer-pytorch