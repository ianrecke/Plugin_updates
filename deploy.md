To deploy the plugin run the following command:
```
sudo docker-compose up --build
```
Once the docker container is created, you should just use: 
```
sudo docker-compose up
```

After that, deploy a `siibra-explorer` dev server as explained in https://github.com/FZJ-INM1-BDA/siibra-explorer/blob/master/README.md but adding the plugin manifest URL so that `siibra-explorer` can load it. In [http.server.dockerfile](./http.server.dockerfile) the port used for the plugin is 6001, so the correct command to run would be:
```
V2_7_PLUGIN_URLS=http://localhost:6001/viewer_plugin/manifest.json node server.js
```
