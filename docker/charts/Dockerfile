FROM python:3.11.5
RUN mkdir /experiment
WORKDIR /experiment
#COPY .python-version ./
COPY plotter.py ./
COPY plot-performance.py ./

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python plotter.py; python plot-performance.py
