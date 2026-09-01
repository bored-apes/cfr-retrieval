"""The vector matrix is memoised per process; an empty load must not be."""

import numpy as np

from cfr import db
from cfr.search import dense


def test_empty_result_is_not_cached(tmp_path):
    """A server started mid-build would otherwise pin an empty matrix forever
    and silently serve no dense results even after the index finished."""
    dense.clear_cache()
    conn = db.connect(tmp_path / "t.db")
    db.init(conn)

    ids, mat = dense.load(conn, "structured")
    assert ids == [] and mat.shape[0] == 0
    assert "structured" not in dense._CACHE

    vec = np.ones(4, dtype=np.float32)
    conn.execute(
        "INSERT INTO documents (doc_id,title,part,section,heading,citation,text) "
        "VALUES ('d','40','262','262.17','h','40 CFR 262.17','body text here')"
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id,doc_id,strategy,ordinal,char_start,char_end,heading_path,text) "
        "VALUES ('c1','d','structured',0,0,4,'p','body')"
    )
    conn.execute(
        "INSERT INTO vectors (chunk_id,strategy,model,dim,vec) VALUES ('c1','structured','m',4,?)",
        (vec.tobytes(),),
    )
    conn.commit()

    ids, mat = dense.load(conn, "structured")
    assert ids == ["c1"], "a later load must pick up vectors written after the first"
    assert mat.shape == (1, 4)
    dense.clear_cache()


def test_vectors_are_normalised_at_load(tmp_path):
    dense.clear_cache()
    conn = db.connect(tmp_path / "t2.db")
    db.init(conn)
    conn.execute(
        "INSERT INTO documents (doc_id,title,part,section,heading,citation,text) "
        "VALUES ('d','40','262','262.17','h','c','t')"
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id,doc_id,strategy,ordinal,char_start,char_end,heading_path,text) "
        "VALUES ('c1','d','structured',0,0,1,'p','t')"
    )
    conn.execute(
        "INSERT INTO vectors (chunk_id,strategy,model,dim,vec) VALUES ('c1','structured','m',3,?)",
        (np.array([3.0, 4.0, 0.0], dtype=np.float32).tobytes(),),
    )
    conn.commit()
    _, mat = dense.load(conn, "structured")
    assert np.isclose(np.linalg.norm(mat[0]), 1.0), "query time is a plain dot product"
    dense.clear_cache()
