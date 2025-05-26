# export_schema.py

from app import create_app
from app.database import db

# from yourapp import create_app, db  # 替换为你实际的 Flask 应用工厂导入路径
from app.models import *  # 已经在 __init__.py 中全部导出
from sqlalchemy.schema import CreateTable


app = create_app()

with app.app_context():
    metadata = db.metadata
    engine = db.engine

    output_file = "schema.sql"
    with open(output_file, "w") as f:
        for table in metadata.sorted_tables:
            create_stmt = str(CreateTable(table).compile(dialect=engine.dialect))
            f.write(create_stmt + ";\n\n")

    print(f"✅ Schema successfully written to {output_file}")
