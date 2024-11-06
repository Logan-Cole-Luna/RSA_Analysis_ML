from Crypto.PublicKey import RSA

def test_rsa_generation():
    try:
        key = RSA.generate(2048, e=65537)
        print("RSA Key generated successfully.")
        print(f"Key Size: {key.size_in_bits()} bits")
        print(f"Public Exponent (e): {key.e}")
    except Exception as ex:
        print(f"Error during RSA key generation: {ex}")

if __name__ == "__main__":
    test_rsa_generation()
