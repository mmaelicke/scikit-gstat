# use Anaconda 3
FROM continuumio/miniconda3:4.7.12

# build some folders 
RUN mkdir /wd
RUN mkdir /wd/skgstat
WORKDIR /wd

# make a data directory
RUN mkdir data
VOLUME ["/wd/data"]

# install some basic packages
# RUN pip install --no-cache-dir ipython jupyter numpy pandas scipy
RUN conda install ipython numpy pandas scipy jupyter numba scikit-learn

# copy the requirements.txt from the scikit-gstat source
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN rm -f requirements.txt

# copy the package itself
ADD skgstat /wd/skgstat

# expose 8888 for jupyter notebooks
EXPOSE 8888

# as soon as we have a demo folder, use this as a starting point
# then the data folder needs to be ln -s
CMD ["sleep 1 && python -m webbrowser http://localhost:8888 &", "jupyter notebook"]

