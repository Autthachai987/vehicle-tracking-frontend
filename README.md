# D.rent_fastapi
 fastapi to connect with cartrack & DTC

 to deploy with raspberryPI you need to make venv for fastapi then
>source D.rent_fastapi/fastapi/bin/activate

now you can run it by
>/D.rent_fastapi fastapi run api_main.py
your api server is now running

next. run vuetify webapp server
>/D.rent_fastapi cd battery_view

>/D.rent_fastapi/battery_view npm run server

If your webapp cannot call any api please check your API server address at
- App.vue
- component BatteryTable.vue
- component FuelTable.vue
also don't forget to change package.json server 
"server": "vite --host 192.168.1.12 --port 3000"
to your RPI ip address
