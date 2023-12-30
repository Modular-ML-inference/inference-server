# Modular Inference Server

The Modular Inference Server has been developed as a part of the [Assist-IoT project](https://assist-iot.eu/). It enables the local inference deployment of a selected model (that can function as a standalone container), the use of flexible configurations, basic format verification and pluggable components. The Modular Inference Server is compatible with Prometheus metric monitoring. 

## Helm chart

The Modular Inference Server enabler ha        s been developed with the assumption that it will be deployed on a Kubernetes cluster with a dedicated Helm chart. To do so, just run `helm install <deployment name> helm-chart`. If you want to deploy multiple Modular Inference Server instances in one Kubernetes cluster, just choose different names for all of the deployments.

To make sure that before that the enabler has been configured properly, check the 2 ConfigMaps that are deployed alongside the enabler. Their names change depending on the name od the deployment (to allow for multiple Modular Inference Server instances to coexist in a Kubernetes cluster while having slightly different configurations). 

The first, which name starts with `flinference-config-map`, serves to flexibly set and change the configuration for the inference component, including the data format received by the gRPC service (as `format.json`), the name, version and input format of the model (as `model.json`), the configuration of the preprocessing as well as postprocessing data transformation pipelines (as `transformation_pipeline.json` and respectively) and the data about both the serialized gRPC service and the specific inferencer to be used (as `setup.json`).

The second config map, which name begins with `fllocalops-config-map` contains the environmental variables necessary to deploy the FL Local Operations instance. Check especially the fields of `REPOSITORY_ADDRESS` (the address of the nearest FL Repository instance). If you change something in the ConfigMap when the enabler is already deployed, destroy the inferenceapp pod to let them recreate with the updated configuration.

## Docker image

You can run `docker compose up --force-recreate --build -d` in your terminal to build a new Docker image.  

## The Modular Inference Server

The Modular Inference Server corresponds to the inferenceapp pod and can function as a standalone. It uses gRPC for lightweight communication. It allows for the configuration setup through the modification of configuration files located in the `configurations` directory (which can also be modified on the fly by changing the values in the `flinference-config-map` and restarting the pod), as well as the addition and subtraction of serialized objects from the (they can be accessed and changed as a Kubernetes volume or downloaded on the fly from the Component Repository in the case of data transformations and models). By default, the inference component accepts data in the form of numerical arrays of any shape and uses a TFLite model to provide lightweight and fast inference. However, it is possible to change the input shape and further details with the use of pluggability.

The inference component is, by default, installed with the rest of the Helm chart. Then it can be accessed through service `fllocaloperationslocal-inferenceapp` on port `50051` according to the specification located in `inference_application/code/proto/extended-inference.proto`.


## Pluggable modules

The the inferenceapp component supports the inference with the TFLite inferencer. However, it is possible to develop custom components for:
  - gRPC service along with the proto and protocompiled files
  - data transformations used for preprocessing and postprocessing
  - inferencer
  - model. 

In order to deploy the image with your custom components through the use of Kubernetes volume, change the `custom_setup` field in `values.yaml` to `True`.

For the extended documentation on how to develop pluggable modules based on some examples, please send me an [email](mailto:karolina.bogacka.dokt@pw.edu.pl)

## Prometheus metric monitoring

The Prometheus metrics are available for scraping on the port `9000` without any additional url path changes in the inferenceapp.


## Licensing

The Modular Inference Server is released under the Apache 2.0 license. 
