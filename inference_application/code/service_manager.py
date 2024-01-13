from inference_application.code.utils import InferenceSetupLoader


class ServiceManager:
    '''
    A manager class, that is responsible for loading the services.
    '''

    def setup_service(self, server):
        """
        Use the InferenceSetupLoader to, step by step, load the relevant modules, the method that binds service to server
        and the finally the relevant servicer.
        """
        setup_loader = InferenceSetupLoader()
        setup_conf = setup_loader.load_setup()
        # Check availability
        setup_loader.check_service_availability(
            setup_conf["service"]["servicer"])
        # Load modules
        setup_loader.load_modules(setup_conf["service"]["modules"])
        # If loaded, load method
        method = setup_loader.load_method(setup_conf["service"]["method"])
        # And finally, servicer
        servicer = setup_loader.load_servicer(
            setup_conf["service"]["servicer"])
        # And link all of them together
        method(servicer(), server)
