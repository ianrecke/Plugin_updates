class Evidence:
    def __init__(self):
        self.evidence = {}
        self.not_evidence_nodes_order = None

    def set(self, nodes_ids, evidence_value):
        for node in nodes_ids:
            self.evidence[node] = evidence_value

    def clear(self, nodes_ids):
        for node in nodes_ids:
            del self.evidence[node]

    def clear_all(self):
        self.evidence = {}

    def get_names(self):
        return list(self.evidence.keys())

    def get(self, node_id):
        try:
            return self.evidence[node_id]
        except KeyError as _:
            return None

    def get_not_evidence_nodes_order(self):
        return self.not_evidence_nodes_order

    def set_not_evidence_nodes_order(self, not_evidence_nodes_order):
        self.not_evidence_nodes_order = not_evidence_nodes_order

    def is_empty(self):
        return not bool(self.evidence)
