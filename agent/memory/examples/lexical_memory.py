"""标准库记忆实验：SQLite/FTS5 + 显式时间更新；没有LLM、向量或后台任务。"""
import json
import sqlite3


def open_memory():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE events(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, raw_json TEXT NOT NULL);
    CREATE TABLE facts(
      id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, project TEXT NOT NULL,
      claim TEXT NOT NULL, keywords TEXT NOT NULL, source_id TEXT NOT NULL,
      valid_from TEXT NOT NULL, valid_to TEXT, revoked INTEGER NOT NULL DEFAULT 0,
      FOREIGN KEY(source_id) REFERENCES events(id));
    CREATE VIRTUAL TABLE fact_search USING fts5(claim, keywords, content='facts', content_rowid='id');
    CREATE TRIGGER fact_insert AFTER INSERT ON facts BEGIN
      INSERT INTO fact_search(rowid,claim,keywords) VALUES(new.id,new.claim,new.keywords);
    END;
    """)
    return db


def remember(db, event_id, tenant, project, claim, keywords, at, supersedes=None):
    """主程序明确给出已确认事实；不声称自动从自然语言抽取或识别冲突。"""
    with db:
        db.execute("INSERT INTO events VALUES(?,?,?)", (
            event_id, tenant, json.dumps({"claim": claim, "observed_at": at}, ensure_ascii=False)))
        if supersedes is not None:
            changed = db.execute(
                "UPDATE facts SET valid_to=? WHERE id=? AND tenant=? AND project=? AND valid_to IS NULL",
                (at, supersedes, tenant, project)).rowcount
            if changed != 1:
                raise ValueError("更新对象不在指定作用域或已被更新")
        cursor = db.execute(
            "INSERT INTO facts(tenant,project,claim,keywords,source_id,valid_from) VALUES(?,?,?,?,?,?)",
            (tenant, project, claim, keywords, event_id, at))
    return cursor.lastrowid


def search(db, tenant, project, query, as_of, k=5):
    # 实验查询使用英文关键词；unicode61不等于中文分词。
    # 每个词作为字面短语，避免把FTS布尔操作符当成用户指令。
    terms = query.split()
    if not terms:
        return []
    literal = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
    rows = db.execute("""
        SELECT f.id,f.claim,f.source_id,bm25(fact_search) AS score
        FROM fact_search JOIN facts f ON f.id=fact_search.rowid
        WHERE fact_search MATCH ? AND f.tenant=? AND f.project=? AND f.revoked=0
          AND f.valid_from<=? AND (f.valid_to IS NULL OR f.valid_to>?)
        ORDER BY score ASC,f.id DESC LIMIT ?
    """, (literal, tenant, project, as_of, as_of, k)).fetchall()
    return [dict(row) for row in rows]


def demo():
    db = open_memory()
    first = remember(db, "e1", "alice", "repo-a", "本项目用pip", "python dependency manager pip",
                     "2026-11-01T00:00:00Z")
    second = remember(db, "e2", "alice", "repo-a", "从今天起本项目用uv", "python dependency manager uv",
                      "2026-11-18T00:00:00Z", supersedes=first)
    remember(db, "e3", "bob", "repo-a", "Bob的同名项目用poetry", "python dependency manager poetry",
             "2026-11-01T00:00:00Z")
    query = "python dependency"
    old = search(db, "alice", "repo-a", query, "2026-11-10T00:00:00Z")
    now = search(db, "alice", "repo-a", query, "2026-11-19T00:00:00Z")
    assert [r["id"] for r in old] == [first]
    assert [r["id"] for r in now] == [second]
    assert search(db, "alice", "repo-b", query, "2026-11-19T00:00:00Z") == []
    assert search(db, "alice", "repo-a", "kubernetes", "2026-11-19T00:00:00Z") == []
    # 逻辑撤销不是物理删除：原始日志/FTS仍在，只是不再返回。
    with db:
        db.execute("UPDATE facts SET revoked=1 WHERE id=?", (second,))
    assert search(db, "alice", "repo-a", query, "2026-11-19T00:00:00Z") == []
    print(json.dumps({"historical": old, "current_before_revoke": now,
                      "after_revoke": [], "checks": "time/scope/empty/revoke passed"},
                     ensure_ascii=False, indent=2))
    db.close()


if __name__ == "__main__":
    demo()
