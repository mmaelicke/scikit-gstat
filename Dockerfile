# use the minimal jupyter notebook
FROM jupyter/minimal-notebook:ad3574d3c5c7

# build tutorials folder 
USER root
RUN mkdir tutorials

# switch back to user
USER $NB_USER

# copy the tutorials content
COPY ./docs/tutorials ./tutorials

# install the latest version 
COPY ./ ./scikit-gstat

# use the latest
RUN cd scikit-gstat && \
    pip install . && \
    cd ..

# the interfaces has two additional 
# optional dependencies: pykrige and gstools
RUN pip install pykrige gstools

# add RISE
RUN conda install -c damianavila82 rise

# switch tu root
USER root

# remove the repo
RUN rm -rf scikit-gstat

# fix permissions
RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# switch back
USER $NB_USER



