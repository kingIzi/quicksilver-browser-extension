import minzoto_ai as m

result = m.detect_mgm("./melaconlique.wav")
print(result)

# import os
# import mux_python
# from mux_python.rest import ApiException
# import minzoto_ai as m

# # Authentication Setup
# configuration = mux_python.Configuration()
# configuration.username = '9f3a1b0b-185a-4535-b1e4-bf29bfcee181'
# configuration.password = 'wDV/omDFIGskEQd/5eJD+GRv+L4X65wo+QBzCuCJIdNCzI0v5iywR3vB24oCNqWHCaCGso+DzR2'

# # API Client Initialization
# assets_api = mux_python.AssetsApi(mux_python.ApiClient(configuration))

# # List Assets
# print("Listing Assets: \n")
# try:
#     list_assets_response = assets_api.list_assets()
#     for asset in list_assets_response.data:
#         print('Asset ID: ' + asset.id)
#         print('Status: ' + asset.status)
#         print('Duration: ' + str(asset.duration))
#         for playback_id in asset.playback_ids or []:
#             print('HLS URL: https://stream.mux.com/' + playback_id.id + '.m3u8')
#             print('Thumbnail: https://image.mux.com/' + playback_id.id + '/thumbnail.jpg')
#         print()
# except ApiException as e:
#     print("Exception when calling AssetsApi->list_assets: %s\n" % e)
