import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import (
    ConfigurationError,
    OperationFailure,
    ServerSelectionTimeoutError,
)


def probar_conexion() -> None:
    load_dotenv()

    uri = os.getenv("MONGODB_URI")

    if not uri:
        print("Error: no se encontró MONGODB_URI en el archivo .env")
        return

    cliente = None

    try:
        cliente = MongoClient(
            uri,
            serverSelectionTimeoutMS=10_000,
        )

        cliente.admin.command("ping")
        print("Conexión exitosa a MongoDB Atlas")

    except OperationFailure:
        print("Error de autenticación: revisa usuario y contraseña.")

    except ServerSelectionTimeoutError:
        print(
            "No se pudo conectar. Revisa Internet y la IP autorizada en Atlas."
        )

    except ConfigurationError as error:
        print(f"Error en la cadena de conexión: {error}")

    except Exception as error:
        print(f"Error inesperado: {error}")

    finally:
        if cliente is not None:
            cliente.close()


if __name__ == "__main__":
    probar_conexion()