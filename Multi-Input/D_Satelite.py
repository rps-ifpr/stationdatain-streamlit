import ee

# Seu ID de cliente (substitua pelo seu ID real)
client_id = '785957889466-i3l8rrgf64lb9lrdigbir3ev4jp6vt1j.apps.googleusercontent.com'

# Autenticação com o ID do cliente
ee.Authenticate(client_id=client_id)

# Inicialização
ee.Initialize()

# Teste de acesso à API
try:
    # Tenta carregar uma imagem de exemplo
    imagem = ee.Image('USGS/SRTMGL1_003')
    print('Acesso à API confirmado! Informações da imagem:')
    print(imagem.getInfo())

except ee.EEException as e:
    print(f'Erro ao acessar a API: {e}')
