#
FROM continuumio/miniconda3:24.1.2-0
# Install Java for syntax checking
RUN apt-get update && apt-get install -y --no-install-recommends default-jre

# Set environment variables
ENV JAVA_HOME /usr/lib/jvm/default-jvm
ENV PATH $JAVA_HOME/bin:$PATH

WORKDIR /usr/src/app

COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate chatty" >> ~/.bashrc
ENV PATH /opt/conda/envs/chatty/bin:$PATH

COPY src ./src
COPY scripts ./scripts

# Expose port
EXPOSE 8501
COPY start.sh start.sh
COPY start_api.sh start_api.sh
RUN chmod +x start.sh
RUN chmod +x start_api.sh
WORKDIR ./src
# CMD ["start.sh"]