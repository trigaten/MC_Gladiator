class Server:
    def __init__(self, fifo:str):
        self.fifo = fifo

    def start(self):
        self.fp = open(self.fifo, "w")

    def execute(self, cmd:str):
        self.fp.write(cmd)