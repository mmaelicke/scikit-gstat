# specify a default Python version
ARG PYTHON_VERSION=3.13

# build from Python
FROM python:${PYTHON_VERSION}
LABEL maintainer="Mirko Mälicke"

# set a user
RUN adduser skguser
USER skguser
WORKDIR /home/skguser

# copy the tutorial
RUN mkdir tutorials
COPY --chown=skguser:skguser ./tutorials ./tutorials

# set the path
ENV PATH="/home/skguser/.local/bin:${PATH}"

# install scikit-gstat
RUN pip install scikit-gstat

# install optional dependencies
RUN pip install gstools pykrige
RUN pip install plotly
RUN pip install rise
RUN pip install jupyter

# open port 8888
EXPOSE 8888

CMD jupyter notebook --ip "0.0.0.0"
