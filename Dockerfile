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

# use a hashed pw if set
ARG PASSWD=nopass
RUN if [ "$PASSWD" = "nopass" ]; then \
    echo "c.NotebookApp.password = u''" >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = u''" >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py; else \
    echo "c.NotebookApp.password = u'$PASSWD'" >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py; \
    fi

# switch tu root
USER root

# remove the repo
RUN rm -rf scikit-gstat

# fix permissions
RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# switch back
USER $NB_USER
