import tenseal as ts
import config
from application.additional.utils import HMSerializer

'''
A file allowing for the generation of a set of new homomorphic keys using the tenseal library.
'''

'''
The parameters named below can be set dynamically according to tenseal documentation.
'''

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 60]
scale_bits = 40
scheme = ts.SCHEME_TYPE.CKKS

'''
The paths below define the names and localization of a secret key and public key necessary for homomorphic encryption.
The secret key here allows for the recreation of the public key, as well as for decrypting the data using the tenseal library.
It is supposed to be located on the Local Operations instances.
The public key allows only for conducting computations on the encrypted data.
It should be located on the Training Collector instance.
'''

secret_name = config.HM_SECRET_FILE
public_name = config.HM_PUBLIC_FILE

context = ts.context(
    scheme,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)
context.generate_galois_keys()
context.global_scale = 2**scale_bits

secret_context = context.serialize(save_secret_key=True)
HMSerializer.write(secret_name, secret_context)

context.make_context_public()
public_context = context.serialize()
HMSerializer.write(public_name, public_context)
