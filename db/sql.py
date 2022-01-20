class Sql:
    def __init__(self,attrs,sel,scan_key,freq):
        self.attributes=attrs
        self.selectivity=sel
        self.scan_key=scan_key
        self.frequency=freq
    
