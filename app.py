import web  # pip install web.py


urls = (
    '/projectch/?', 'application.controllers.projectch.Projectch',
)


app = web.application(urls, globals())


if __name__ == "__main__":
    web.config.debug = False
    app.run()
