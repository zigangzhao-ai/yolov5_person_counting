from qiniu import put_file, Auth
access_key = "mk76AFsdfjskadjkdsfkdjf4eeDgBpe"
secret_key = "WrfdgdfegfdgsadfW89WasdfsadfwnSyND"
bucket_name = "qigeming"
q = Auth(access_key, secret_key)
token = q.upload_token(bucket_name, key)
ret, info = put_file(token, key, sp)
return "{}{}?v={}".format("http://p.youradminhost.cn/", key, m)