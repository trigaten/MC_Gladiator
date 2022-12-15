class Server:
    def __init__(fifo:str):
        self.fifo = fifo

    def start(self):
        self.fp = open("urfifo", "w")

    def execute(self, cmd:str):
        self.fp.write(cmd)