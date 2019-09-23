# getdata chen_begin
class base_graph():
    globals = None
    nodes_pt = None
    nodes_gt = None
    edges = None
    receivers = None
    senders = None
    def __init__(self, nodes_pt_,nodes_gt_):
        self.nodes_pt = nodes_pt_
        self.nodes_gt = nodes_gt_
# getdata chen_end