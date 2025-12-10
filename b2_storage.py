"""
🔒 B2 Storage Module - FINAL
Integração oficial com Backblaze B2 (Private Bucket)
Funções:
- Upload
- Signed URL privada
- Listagem
- Delete
"""

import os
import logging
from b2sdk.v2 import InMemoryAccountInfo, B2Api

logger = logging.getLogger("B2Storage")


# ------------------------------------------------------------------------------------
# CLIENTE B2
# ------------------------------------------------------------------------------------

def get_b2_api():
    """Autentica no B2 usando variáveis de ambiente."""
    try:
        key_id = os.getenv("B2_KEY_ID")
        application_key = os.getenv("B2_APPLICATION_KEY")
        bucket_name = os.getenv("B2_BUCKET_NAME")

        if not all([key_id, application_key, bucket_name]):
            raise ValueError("Variáveis B2 não configuradas.")

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)

        b2_api.authorize_account("production", key_id, application_key)
        bucket = b2_api.get_bucket_by_name(bucket_name)

        return b2_api, bucket

    except Exception as e:
        logger.error(f"❌ Erro ao conectar no B2: {e}")
        raise e


# ------------------------------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------------------------------

def upload_file(file_path: str, file_name: str):
    """Upload direto para bucket privado."""
    try:
        _, bucket = get_b2_api()

        bucket.upload_local_file(
            local_file=file_path,
            file_name=file_name
        )

        logger.info(f"📤 Upload concluído: {file_name}")
        return file_name

    except Exception as e:
        logger.error(f"❌ Falha no upload: {e}")
        raise e


# ------------------------------------------------------------------------------------
# SIGNED URL PRIVADA
# ------------------------------------------------------------------------------------

def generate_signed_download_url(file_name: str, expires_in: int = 3600):
    """Cria Signed URL via download authorization (privada)."""
    try:
        b2_api, bucket = get_b2_api()

        auth_token = bucket.get_download_authorization(
            file_name_prefix=file_name,
            valid_duration_in_seconds=expires_in
        )

        base = bucket.get_download_url(file_name)
        signed_url = f"{base}?Authorization={auth_token}"

        return signed_url

    except Exception as e:
        logger.error(f"❌ Erro ao gerar Signed URL: {e}")
        return None


# ------------------------------------------------------------------------------------
# DELETE
# ------------------------------------------------------------------------------------

def delete_file(file_name: str):
    try:
        _, bucket = get_b2_api()
        version = bucket.get_file_info_by_name(file_name)

        if version:
            bucket.delete_file_version(version.id_, file_name)
            logger.info(f"🗑️ Deletado: {file_name}")
        else:
            logger.warning(f"Arquivo não encontrado: {file_name}")

    except Exception as e:
        logger.error(f"❌ Erro ao deletar arquivo B2: {e}")
        raise e


# ------------------------------------------------------------------------------------
# LISTAGEM
# ------------------------------------------------------------------------------------

def list_files(prefix: str = "") -> list:
    """Lista arquivos no bucket."""
    try:
        _, bucket = get_b2_api()

        result = []
        for version, _ in bucket.ls(folder_to_list=prefix, recursive=True):
            result.append(version.file_name)

        return result

    except Exception as e:
        logger.error(f"❌ Erro ao listar arquivos B2: {e}")
        return []
