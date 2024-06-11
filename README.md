# Modular Inference Server

The Modular Inference Server has been developed as a part of the [ASSIST-IoT project](https://assist-iot.eu/). It enables the local inference deployment of a selected model (that can function as a standalone container), the use of flexible configurations, basic format verification and pluggable components. The Modular Inference Server is compatible with Prometheus metric monitoring. Modular Inference Server forms the basis for the paper ["Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum"](https://www.mdpi.com/2079-9292/13/10/1888). 

## Helm chart

The Modular Inference Server enabler has been developed with the assumption that it will be deployed on a Kubernetes cluster with a dedicated Helm chart. To do so, just run `helm install <deployment name> helm-chart`. If you want to deploy multiple Modular Inference Server instances in one Kubernetes cluster, just choose different names for all of the deployments.

To make sure that before that the enabler has been configured properly, check the 2 ConfigMaps that are deployed alongside the enabler. Their names change depending on the name od the deployment (to allow for multiple Modular Inference Server instances to coexist in a Kubernetes cluster while having slightly different configurations). 

The first, which name starts with `ml-inf-pipeline-config-map-`, serves to flexibly set and change the configuration for the inference component, including the data format received by the gRPC service (as `format.json`), the name, version and input format of the model (as `model.json`), the configuration of the preprocessing as well as postprocessing data transformation pipelines (as `transformation_pipeline.json` and respectively) and the data about both the serialized gRPC service and the specific inferencer to be used (as `setup.json`).

The second config map, which name begins with `ml-inf-global-values-config-map-` contains the environmental variables necessary to deploy the FL Local Operations instance. Check especially the fields of `REPOSITORY_ADDRESS` (the address of the nearest FL Repository instance). If you change something in the ConfigMap when the enabler is already deployed, destroy the inferenceapp pod to let them recreate with the updated configuration.

## Docker image

You can run `docker compose up --force-recreate --build -d` in your terminal to build a new Docker image. As the `Dockerfile` and `docker-compose.yml` have already been constructed for multi-arch deployment, in order to build the relevant multi-arch image, just run `docker buildx bake -f docker-compose.yml --set *.platform=linux/amd64,linux/arm64 --push`. 
 

## The Modular Inference Server

The Modular Inference Server corresponds to the inferenceapp pod and can function as a standalone. It uses gRPC streaming for lightweight communication. It allows for the configuration setup through the modification of configuration files located in the `configurations` directory (which can also be modified on the fly by changing the values in the `flinference-config-map` and restarting the pod), as well as the addition and subtraction of serialized objects from the (they can be accessed and changed as a Kubernetes volume or downloaded on the fly from the Component Repository in the case of data transformations and models). By default, the inference component accepts data in the form of numerical arrays of any shape and uses a TFLite model to provide lightweight and fast inference. However, it is possible to change the input shape and further details with the use of pluggability.

The inference component is, by default, installed with the rest of the Helm chart. Then it can be accessed through the Kubernetes service of the app on the port `50051` according to the specification located in `inference_application/code/proto/extended-inference.proto`.

## Use case configurations

Use case configurations that allow one to recreate the Modular Inference Server deployment for fall detection and scratch detection are available in the `use_case_configurations` folder under the relevant names.  

## Pluggable modules

The the inferenceapp component supports the inference with the TFLite and Torch-RCNN inferencer. However, it is possible to develop custom components for:
  - gRPC service along with the proto and protocompiled files
  - data transformations used for preprocessing and postprocessing
  - inferencer
  - model. 

In order to deploy the image with your custom components through the use of Kubernetes volume, change the `custom_setup` field in `values.yaml` to `True`. Here, some instructions on how to develop 2 of the pluggable modules will follow. The other two, `inferencers` and `services`, will behave similarly.

### Model

A new FL model can be saved either in a format ready for FL inference in TFLite. For TFLite, the method used
should be:
   
```python
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

Then, the file should be compressed to a ZIP format in order to save
space, for example using this snippet of code:

```python
with zipfile.ZipFile('tflite_test_model.zip', 'w') as f:       
  upper_dir = pathlib.Path("model/")
  for file in upper_dir.rglob("*"):
    f.write(file)
```

They can be uploaded using the Swagger API of the Component Repository, by first creating the metadata of the model and then uploading a file by updating the object for a given metadata. Then, the `model_name` and `model_version` in the configuration should be updated accordingly.

### Data Transformation

We will be demonstrating how to construct and configure the loading of a
data transformation for the inference module.

**Attention**: To do so, first make sure that the environment you’re
using has a Python version compatible with the inference module, that
is, 3.11.4. Otherwise, you may encounter problems related to magic numbers.

First, let’s design the transformation. Here is a sample data
transformation:

```python
   from data_transformation.transformation import DataTransformation
   from datamodels.models import MachineCapabilities


   import numpy as np

   class BasicDimensionExpansionTransformation(DataTransformation):
       import numpy as np

       id = "basic-expand-dimensions"
       description = """Basically a wrapper around numpy.expand_dims. 
       Expands the shape of the array by inserting a new axis, that will appear at the axis position in expanded array shape"""
       parameter_types = {"axis": int}
       default_values = {"axis": 0}
       outputs = [np.ndarray]
       needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

       def set_parameters(self, parameters):
           self.params = parameters

       def get_parameters(self):
           return self.params

       def transform_data(self, data):
           data = np.array(data)
           return np.expand_dims(data, axis=self.params["axis"])

       def transform_format(self, format):
           if "numerical" in format["data_types"]:
               axis = self.params["axis"]
               format["data_types"]["numerical"]["size"].insert(axis, 1)
           return format
```

The new data transformation class should be a subclass of the abstract
DataTransformation class from the data_transformation module. It should
have a unique id, a description of purpose, a dictionary of parameter
types, a dictionary of default values, a list of output types and a
MachineCapabilities object that expresses what needs to be present in
the Docker container/on the machine to run this transformation.

If you have this transformation ready, you should put it, for example,
in the ``inference_application/custom`` directory in the FL Local
Operations repository and use a different file to properly serialize the
modules. Like this:

```python
   with zipfile.PyZipFile("inference_application.custom.expansion.zip", mode="w") as zip_pkg:
        zip_pkg.writepy("inference_application/custom/expansion.py")

   with open('inference_application.custom.expansion.pkl', 'wb') as f:
       dill.dump(BasicDimensionExpansionTransformation,f)
```

For serializing this data, both zipimport and dill were used to make
sure that even the most complicated transformations will be possible to
load. Just remember to name the files according to the paths to the
modules you would like to serialize (just replace the “/” with the “.”).

Then, you can either zip the two resulting files and upload them to the
Component Repository as a transformation, or place the files in the
``inference_application/local_cache/transformations`` directory, either
by building a new image or deploying the Helm chart with the
``customSetup`` field marked to true in ``values.yaml`` file for the
inference application and using ``kubectl cp`` to place the files.

Finally, you can apprioprately change the inference application
configuration to use that specific transformation with selected
parameters. You can do it by modifying the appropriate ConfigMap.

```json
   [
       {
           "id": "inference_application.custom.basic_norm",
           "parameters": {

           }
       },
       {
           "id": "inference_application.custom.expansion",
           "parameters": {
               "axis": 0

           }
       }
   ]
```

If you just want to reuse an existing transformation, it’s enough to
only modify the configuration. 

## Prometheus metric monitoring

The Prometheus metrics are available for scraping on the port `9000` without any additional url path changes in the inferenceapp.

## Citation

If you found the Modular Inference Server useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```
Bogacka, K.; Sowiński, P.; Danilenka, A.; Biot, F.M.; Wasielewska-Michniewska, K.; Ganzha, M.; Paprzycki, M.; Palau, C.E. Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum. Electronics 2024, 13, 1888. https://doi.org/10.3390/electronics13101888 
```

```bibtex
@Article{electronics13101888,
AUTHOR = {Bogacka, Karolina and Sowiński, Piotr and Danilenka, Anastasiya and Biot, Francisco Mahedero and Wasielewska-Michniewska, Katarzyna and Ganzha, Maria and Paprzycki, Marcin and Palau, Carlos E.},
TITLE = {Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {10},
ARTICLE-NUMBER = {1888},
URL = {https://www.mdpi.com/2079-9292/13/10/1888},
ISSN = {2079-9292},
ABSTRACT = {Currently, deploying machine learning workloads in the Cloud–Edge–IoT continuum is challenging due to the wide variety of available hardware platforms, stringent performance requirements, and the heterogeneity of the workloads themselves. To alleviate this, a novel, flexible approach for machine learning inference is introduced, which is suitable for deployment in diverse environments—including edge devices. The proposed solution has a modular design and is compatible with a wide range of user-defined machine learning pipelines. To improve energy efficiency and scalability, a high-performance communication protocol for inference is propounded, along with a scale-out mechanism based on a load balancer. The inference service plugs into the ASSIST-IoT reference architecture, thus taking advantage of its other components. The solution was evaluated in two scenarios closely emulating real-life use cases, with demanding workloads and requirements constituting several different deployment scenarios. The results from the evaluation show that the proposed software meets the high throughput and low latency of inference requirements of the use cases while effectively adapting to the available hardware. The code and documentation, in addition to the data used in the evaluation, were open-sourced to foster adoption of the solution.},
DOI = {10.3390/electronics13101888}
}

```

## Author

[Karolina Bogacka](https://orcid.org/0000-0002-7109-891X) ([GitHub](https://github.com/Karolina-Bogacka))

## Licensing

The Modular Inference Server is released under the Apache 2.0 license. 
