import subprocess


class Server:
    def __init__(self, screen_name:str):
        self.screen_name = screen_name

    def execute(self, cmd:str):
        subprocess.call(["screen", "-p", "0", "-S", self.screen_name, "-X", "eval", f"stuff '{cmd}^M'"])

if __name__ == "__main__":
    s = Server("mc")
    s.execute("tp human ~ 100 ~")