from service.server import APP, PORT

if __name__ == '__main__':
    from wsgiref import simple_server
    httpd = simple_server.make_server('0.0.0.0', PORT, APP)
    httpd.serve_forever()

    
