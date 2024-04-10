import itertools

from lshashpy3 import LSHash

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


class WideHash(LSHash):
    """
    A class to perform approximate nearest neighbour search in high dimensions.
    It stores a number of hash tables (as per LSHash) and, when queried with a vector
    find vectors that fall into the same bucket in any of the hash tables.
    It then 
    """

    def query(self, query_point, num_results=None, distance_func=None,
              max_width: int = 0):
        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.
        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        :param max_width:
            (optional) the number of bits to switch in the binary hash in
            order to make the search wider. If 0, only candidates that fall
            in the exact bucket as query_point will be returned, if 1,
            also those which fall in a bucket with binar-distance 1 from
            query_points's bucket will be returned, and so on. To avoid
            computation taking too long, it will be capped at 3
        """

        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hash_tables):
                b_hash = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = LSHash.hamming_dist(key, b_hash)
                    if distance < 2:
                        candidates.update(table.get_list(key))

            d_func = LSHash.euclidean_dist_square

        else:

            if distance_func == "euclidean":
                d_func = LSHash.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LSHash.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LSHash.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LSHash.cosine_dist
            elif distance_func == "l1norm":
                d_func = LSHash.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")

            for i, table in enumerate(self.hash_tables):
                b_hash = self._hash(self.uniform_planes[i], query_point)
                candidates.update(table.get_list(b_hash))
                if max_width > 0:
                    rang = list(range(self.hash_size))
                    for width in range(1, min(max_width, 3)+1):
                        for bits_to_change in itertools.combinations(rang,
                                                                     width):
                            althash = "".join([str(1 - int(bit))
                                               if i in bits_to_change
                                               else bit
                                               for i, bit in
                                               enumerate(b_hash)])
                            candidates.update(table.get_list(althash))

        # rank candidates by distance function
        candidates = [(ix, d_func(query_point, self._as_np_array(ix)))
                      for ix in candidates]
        candidates = sorted(candidates, key=lambda x: x[1])
        

        return candidates[:num_results] if num_results else candidates
