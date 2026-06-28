import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from rag.retriever import Retriever

r = Retriever()
r.index_documents([
    "Python использует отступы для блоков кода",
    "Docker позволяет запускать приложения в контейнерах",
    "Git хранит историю изменений в репозитории",
])

print(r.retrieve("как работают контейнеры?"))