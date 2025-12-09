"""
🔒 B2 Storage Module - Integração Backblaze B2 (Secure Signed URLs)
Gerencia uploads, downloads e URLs temporárias privadas usando b2sdk.v2.
"""

import os
import logging
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from botocore.exceptions import ClientError

# Configuração de Logs
logger = logging.getLogger("B2Storage")

def get_b2_api():
    """
    Inicializa e autentica o cliente B2Api usando variáveis de ambiente.
    Retorna a instância autenticada da API e o objeto bucket.
    """
    try:
        key_id = os.getenv("B2_KEY_ID")
        application_key = os.getenv("B2_APPLICATION_KEY")
        bucket_name = os.getenv("B2_BUCKET_NAME")

        if not all([key_id, application_key, bucket_name]):
            raise ValueError("❌ Variáveis de ambiente do Backblaze B2 não configuradas.")

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", key_id, application_key)
        
        bucket = b2_api.get_bucket_by_name(bucket_name)
        return b2_api, bucket

    except Exception as e:
        logger.error(f"❌ Erro ao conectar ao Backblaze B2: {e}")
        raise e

def upload_file(file_path: str, file_name: str) -> str:
    """
    Faz upload de um arquivo local para o bucket privado.
    Sobrescreve se já existir. Retorna o nome do arquivo no bucket.
    """
    try:
        _, bucket = get_b2_api()
        
        logger.info(f"📤 Iniciando upload para B2: {file_name}")
        
        # Upload com stream para eficiência
        bucket.upload_local_file(
            local_file=file_path,
            file_name=file_name
        )
        
        logger.info(f"✅ Upload concluído: {file_name}")
        return file_name

    except Exception as e:
        logger.error(f"❌ Falha no upload para B2: {e}")
        raise e

def generate_signed_download_url(file_name: str, expires_in: int = 3600) -> str:
    """
    Gera uma URL assinada (privada) para download temporário.
    A URL expira em 'expires_in' segundos (Padrão: 1h).
    """
    try:
        _, bucket = get_b2_api()
        
        # Obtém token de autorização para o arquivo específico
        # O prefixo vazio "" significa que o token é válido para este aquivo
        # duration é em segundos
        auth_token = bucket.get_download_authorization(
            file_name_prefix=file_name,
            valid_duration_in_seconds=expires_in
        )
        
        # Constrói a URL final usando a API de download do B2
        # Formato: https://f002.backblazeb2.com/file/<bucket_name>/<file_name>?Authorization=<token>
        # Precisamos da URL base do download
        download_url_base = bucket.get_download_url(file_name)
        
        signed_url = f"{download_url_base}?Authorization={auth_token}"
        
        return signed_url

    except Exception as e:
        logger.error(f"❌ Erro ao gerar Signed URL B2: {e}")
        return None

def delete_file(file_name: str) -> None:
    """
    Remove um arquivo do bucket pelo nome/caminho.
    """
    try:
        _, bucket = get_b2_api()
        
        # Primeiro precisamos encontrar o file_id da versão mais recente
        file_version = bucket.get_file_info_by_name(file_name)
        
        if file_version:
            bucket.delete_file_version(file_version.id_, file_name)
            logger.info(f"🗑️ Arquivo deletado do B2: {file_name}")
        else:
            logger.warning(f"⚠️ Arquivo não encontrado para deleção: {file_name}")

    except Exception as e:
        logger.error(f"❌ Erro ao deletar arquivo B2: {e}")
        raise e

def list_files(prefix: str = "") -> list:
    """
    Lista arquivos no bucket com um prefixo opcional.
    Retorna lista de nomes de arquivos.
    """
    try:
        _, bucket = get_b2_api()
        
        files = []
        # ls retorna um generator de (file_version, folder_name)
        for output in bucket.ls(folder_to_list=prefix, recursive=True):
            file_version, _ = output
            # file_version é um objeto FileVersionInfo
            files.append(file_version.file_name)
            
        return files

    except Exception as e:
        logger.error(f"❌ Erro ao listar arquivos B2: {e}")
        return []
