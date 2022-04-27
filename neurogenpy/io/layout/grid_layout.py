from .graph_layout import GraphLayout


class GridLayout(GraphLayout):
    """
    Grid layout class.
    """

    def run(self, env='igraph'):
        """
        Calculates the layout for the graph with the grid algorithm.

        Parameters
        ----------
        env : str, default='igraph'
            Environment used to calculate the layout.

        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.

        Raises
        ------
        ValueError
            If the environment selected is not supported.
        """

        if env == 'igraph':
            return self._run_igraph('grid')
        else:
            raise ValueError(f'{env} environment is not supported.')
