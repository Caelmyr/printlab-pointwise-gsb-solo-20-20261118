import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .pixels import json_safe, png_bytes, thumbnail


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identifier(prefix):
    return prefix + '_' + uuid.uuid4().hex[:16]


def safe_label(value, maximum=120):
    label = str(value or '').strip()
    if not label:
        raise ValueError('名称不能为空')
    if len(label) > maximum:
        raise ValueError(f'名称最多{maximum}个字符')
    if any(ord(char) < 32 for char in label):
        raise ValueError('名称包含控制字符')
    return label


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / 'blobs'
        self.blobs.mkdir(exist_ok=True)
        self.staging = self.root / 'staging'
        self.staging.mkdir(exist_ok=True)
        self.database = self.root / 'printlab.sqlite3'
        self.lock = threading.RLock()
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=20000')
        return connection

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            connection = self.connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def initialize(self):
        with self.connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT NOT NULL, dpi REAL NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    settings TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    name TEXT NOT NULL, kind TEXT NOT NULL,
                    blob TEXT NOT NULL, preview_blob TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL,
                    channels INTEGER NOT NULL, metadata TEXT NOT NULL,
                    parent_id TEXT REFERENCES assets(id),
                    operation_id TEXT, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assets_job ON assets(job_id, created_at);
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    operation TEXT NOT NULL, source_id TEXT NOT NULL REFERENCES assets(id),
                    reference_id TEXT REFERENCES assets(id), parameters TEXT NOT NULL,
                    summary TEXT NOT NULL, tables_data TEXT NOT NULL,
                    created_at TEXT NOT NULL, elapsed_ms REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS operations_job ON operations(job_id, created_at);
                CREATE TABLE IF NOT EXISTS regions (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
                    name TEXT NOT NULL, geometry TEXT NOT NULL,
                    role TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS regions_asset ON regions(asset_id);
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    kind TEXT NOT NULL, name TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    asset_id TEXT REFERENCES assets(id), text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT, event TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
            ''')

    def serialize(self, value):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    def event(self, connection, job_id, event, payload):
        connection.execute('INSERT INTO events(job_id,event,payload,created_at) VALUES(?,?,?,?)', (job_id, event, self.serialize(payload), utc_now()))

    def job_row(self, row):
        result = dict(row)
        result['settings'] = json.loads(result['settings'])
        return result

    def asset_row(self, row):
        result = dict(row)
        result['metadata'] = json.loads(result['metadata'])
        result.pop('blob', None)
        result.pop('preview_blob', None)
        result['preview_url'] = f'/api/assets/{result["id"]}/preview'
        result['download_url'] = f'/api/assets/{result["id"]}/download'
        return result

    def operation_row(self, row):
        result = dict(row)
        result['parameters'] = json.loads(result['parameters'])
        result['summary'] = json.loads(result['summary'])
        result['tables'] = json.loads(result.pop('tables_data'))
        return result

    def require_job(self, connection, job_id):
        row = connection.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError('印刷作业不存在')
        return row

    def require_asset(self, connection, asset_id):
        row = connection.execute('SELECT * FROM assets WHERE id=?', (asset_id,)).fetchone()
        if row is None:
            raise KeyError('图像不存在')
        return row

    def touch(self, connection, job_id):
        connection.execute('UPDATE jobs SET updated_at=?, revision=revision+1 WHERE id=?', (utc_now(), job_id))

    def create_job(self, name, dpi=300, description=''):
        title = safe_label(name)
        if len(str(description)) > 4000:
            raise ValueError('作业说明过长')
        resolution = float(dpi)
        if not np.isfinite(resolution) or not 36 <= resolution <= 2400:
            raise ValueError('分辨率需要36到2400 DPI')
        job_id = identifier('job')
        now = utc_now()
        with self.transaction() as connection:
            connection.execute('INSERT INTO jobs(id,name,description,dpi,created_at,updated_at) VALUES(?,?,?,?,?,?)', (job_id, title, str(description), resolution, now, now))
            self.event(connection, job_id, 'job_created', {'name': title})
        return self.get_job(job_id)

    def list_jobs(self):
        with self.connect() as connection:
            rows = connection.execute('SELECT j.*, (SELECT COUNT(*) FROM assets a WHERE a.job_id=j.id) asset_count, (SELECT COUNT(*) FROM operations o WHERE o.job_id=j.id) operation_count FROM jobs j ORDER BY updated_at DESC').fetchall()
        return [self.job_row(row) for row in rows]

    def get_job(self, job_id):
        with self.connect() as connection:
            row = self.require_job(connection, job_id)
            result = self.job_row(row)
            result['assets'] = [self.asset_row(item) for item in connection.execute('SELECT * FROM assets WHERE job_id=? ORDER BY created_at,id', (job_id,))]
            result['operations'] = [self.operation_row(item) for item in connection.execute('SELECT * FROM operations WHERE job_id=? ORDER BY created_at DESC,id', (job_id,))]
            result['profiles'] = [self.profile_row(item) for item in connection.execute('SELECT * FROM profiles WHERE job_id=? ORDER BY created_at', (job_id,))]
            result['notes'] = [dict(item) for item in connection.execute('SELECT * FROM notes WHERE job_id=? ORDER BY created_at DESC', (job_id,))]
        return result

    def update_job(self, job_id, payload):
        with self.transaction() as connection:
            current = self.require_job(connection, job_id)
            if 'revision' in payload and int(payload['revision']) != current['revision']:
                raise ValueError('作业已变更，请刷新后再保存')
            name = safe_label(payload.get('name', current['name']))
            description = str(payload.get('description', current['description']))
            if len(description) > 4000:
                raise ValueError('作业说明过长')
            dpi = float(payload.get('dpi', current['dpi']))
            if not np.isfinite(dpi) or not 36 <= dpi <= 2400:
                raise ValueError('分辨率需要36到2400 DPI')
            settings = payload.get('settings', json.loads(current['settings']))
            if not isinstance(settings, dict):
                raise ValueError('作业设置必须为对象')
            connection.execute('UPDATE jobs SET name=?,description=?,dpi=?,settings=? WHERE id=?', (name, description, dpi, self.serialize(settings), job_id))
            self.touch(connection, job_id)
            self.event(connection, job_id, 'job_updated', {'name': name, 'dpi': dpi})
        return self.get_job(job_id)

    def write_blob(self, data, extension):
        digest = hashlib.sha256(data).hexdigest()
        filename = digest + '.' + extension
        destination = self.blobs / filename
        if not destination.exists():
            temporary = self.staging / (uuid.uuid4().hex + '.part')
            try:
                with temporary.open('xb') as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return filename

    def write_array(self, array):
        values = np.ascontiguousarray(array, dtype=np.float32)
        if values.ndim not in (2, 3):
            raise ValueError('图像维度不支持')
        if not np.all(np.isfinite(values)):
            raise ValueError('图像包含无效像素')
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        return self.write_blob(stream.getvalue(), 'npy')

    def read_blob(self, name):
        if not re.fullmatch(r'[a-f0-9]{64}\.(npy|png|bin)', name):
            raise ValueError('无效的内容地址')
        path = self.blobs / name
        if not path.is_file():
            raise RuntimeError('图像内容丢失')
        return path.read_bytes()

    def asset_record(self, job_id, name, kind, array, metadata=None, parent_id=None, operation_id=None, preview=None):
        values = np.asarray(array, dtype=np.float32)
        channels = values.shape[2] if values.ndim == 3 else 1
        if channels not in (1, 3, 4):
            raise ValueError('图像通道数不支持')
        image_preview = preview if preview is not None else values
        if channels == 4 and preview is None:
            image_preview = (1 - values[:, :, :3]) * (1 - values[:, :, 3:4])
        return {
            'id': identifier('asset'),
            'job_id': job_id,
            'name': safe_label(name),
            'kind': safe_label(kind, 60),
            'blob': self.write_array(values),
            'preview_blob': self.write_blob(png_bytes(thumbnail(image_preview)), 'png'),
            'width': values.shape[1],
            'height': values.shape[0],
            'channels': channels,
            'metadata': self.serialize(metadata or {}),
            'parent_id': parent_id,
            'operation_id': operation_id,
            'created_at': utc_now(),
        }

    def insert_asset(self, connection, record):
        keys = list(record)
        sql = 'INSERT INTO assets(' + ','.join(keys) + ') VALUES(' + ','.join('?' for _ in keys) + ')'
        connection.execute(sql, [record[key] for key in keys])

    def add_asset(self, job_id, name, kind, array, metadata=None, parent_id=None, preview=None):
        with self.connect() as connection:
            self.require_job(connection, job_id)
            if parent_id:
                parent = self.require_asset(connection, parent_id)
                if parent['job_id'] != job_id:
                    raise ValueError('父图像不属于该作业')
        record = self.asset_record(job_id, name, kind, array, metadata, parent_id, preview=preview)
        with self.transaction() as connection:
            self.require_job(connection, job_id)
            self.insert_asset(connection, record)
            self.touch(connection, job_id)
            self.event(connection, job_id, 'asset_added', {'asset_id': record['id'], 'kind': kind})
        return self.get_asset(record['id'])

    def get_asset(self, asset_id):
        with self.connect() as connection:
            row = self.require_asset(connection, asset_id)
            result = self.asset_row(row)
            result['regions'] = [self.region_row(item) for item in connection.execute('SELECT * FROM regions WHERE asset_id=? ORDER BY created_at', (asset_id,))]
        return result

    def load_array(self, asset_id):
        with self.connect() as connection:
            row = self.require_asset(connection, asset_id)
            payload = self.read_blob(row['blob'])
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        return np.asarray(array, dtype=np.float32)

    def preview(self, asset_id):
        with self.connect() as connection:
            row = self.require_asset(connection, asset_id)
            return self.read_blob(row['preview_blob'])

    def save_operation(self, job_id, operation, source_id, reference_id, parameters, result, elapsed_ms):
        summary = result.get('summary', {})
        tables = result.get('tables', {})
        images = result.get('images', {})
        if not isinstance(summary, dict) or not isinstance(tables, dict) or not isinstance(images, dict):
            raise ValueError('处理结果格式无效')
        with self.connect() as connection:
            source = self.require_asset(connection, source_id)
            if source['job_id'] != job_id:
                raise ValueError('源图像不属于该作业')
            if reference_id:
                reference = self.require_asset(connection, reference_id)
                if reference['job_id'] != job_id:
                    raise ValueError('参考图像不属于该作业')
        operation_id = identifier('op')
        created = utc_now()
        records = []
        output_kinds = result.get('kinds', {})
        for label, array in images.items():
            values = np.asarray(array)
            inferred = 'cmyk' if values.ndim == 3 and values.shape[2] == 4 else 'rgb' if values.ndim == 3 else 'gray'
            kind = output_kinds.get(label, inferred)
            metadata = {'operation': operation, 'result_key': label, 'source_id': source_id, 'reference_id': reference_id}
            metadata.update(result.get('metadata', {}).get(label, {}))
            records.append(self.asset_record(job_id, label, kind, values, metadata, source_id, operation_id))
        with self.transaction() as connection:
            connection.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?)', (operation_id, job_id, operation, source_id, reference_id, self.serialize(parameters), self.serialize(summary), self.serialize(tables), created, float(elapsed_ms)))
            for record in records:
                self.insert_asset(connection, record)
            self.touch(connection, job_id)
            self.event(connection, job_id, 'operation_completed', {'operation_id': operation_id, 'operation': operation, 'output_ids': [r['id'] for r in records]})
        return self.get_operation(operation_id)

    def get_operation(self, operation_id):
        with self.connect() as connection:
            row = connection.execute('SELECT * FROM operations WHERE id=?', (operation_id,)).fetchone()
            if row is None:
                raise KeyError('处理记录不存在')
            result = self.operation_row(row)
            result['outputs'] = [self.asset_row(item) for item in connection.execute('SELECT * FROM assets WHERE operation_id=? ORDER BY created_at,id', (operation_id,))]
        return result

    def region_row(self, row):
        result = dict(row)
        result['geometry'] = json.loads(result['geometry'])
        return result

    def add_region(self, asset_id, name, geometry, role='measure'):
        from .regions import validate_geometry
        with self.transaction() as connection:
            asset = self.require_asset(connection, asset_id)
            checked = validate_geometry(geometry, (asset['height'], asset['width']))
            if role not in ('measure', 'exclude', 'protect', 'sample'):
                raise ValueError('区域用途不支持')
            region_id = identifier('roi')
            now = utc_now()
            connection.execute('INSERT INTO regions VALUES(?,?,?,?,?,?,?)', (region_id, asset_id, safe_label(name), self.serialize(checked), role, now, now))
            self.touch(connection, asset['job_id'])
            self.event(connection, asset['job_id'], 'region_added', {'asset_id': asset_id, 'region_id': region_id, 'role': role})
            row = connection.execute('SELECT * FROM regions WHERE id=?', (region_id,)).fetchone()
        return self.region_row(row)

    def delete_region(self, region_id):
        with self.transaction() as connection:
            row = connection.execute('SELECT r.*,a.job_id FROM regions r JOIN assets a ON a.id=r.asset_id WHERE r.id=?', (region_id,)).fetchone()
            if row is None:
                raise KeyError('区域不存在')
            connection.execute('DELETE FROM regions WHERE id=?', (region_id,))
            self.touch(connection, row['job_id'])
            self.event(connection, row['job_id'], 'region_deleted', {'region_id': region_id})
        return {'deleted': region_id}

    def profile_row(self, row):
        result = dict(row)
        result['payload'] = json.loads(result['payload'])
        return result

    def save_profile(self, job_id, kind, name, payload, profile_id=None):
        from .profiles import validate_profile
        checked = validate_profile(kind, payload)
        title = safe_label(name)
        now = utc_now()
        with self.transaction() as connection:
            self.require_job(connection, job_id)
            if profile_id:
                previous = connection.execute('SELECT * FROM profiles WHERE id=? AND job_id=?', (profile_id, job_id)).fetchone()
                if previous is None:
                    raise KeyError('配置不存在')
                if previous['kind'] != kind:
                    raise ValueError('不能更改配置种类')
                connection.execute('UPDATE profiles SET name=?,payload=?,updated_at=? WHERE id=?', (title, self.serialize(checked), now, profile_id))
            else:
                profile_id = identifier('profile')
                connection.execute('INSERT INTO profiles VALUES(?,?,?,?,?,?,?)', (profile_id, job_id, kind, title, self.serialize(checked), now, now))
            self.touch(connection, job_id)
            self.event(connection, job_id, 'profile_saved', {'profile_id': profile_id, 'kind': kind})
            row = connection.execute('SELECT * FROM profiles WHERE id=?', (profile_id,)).fetchone()
        return self.profile_row(row)

    def get_profile(self, profile_id):
        with self.connect() as connection:
            row = connection.execute('SELECT * FROM profiles WHERE id=?', (profile_id,)).fetchone()
            if row is None:
                raise KeyError('配置不存在')
        return self.profile_row(row)

    def add_note(self, job_id, text, asset_id=None):
        body = str(text).strip()
        if not body or len(body) > 4000:
            raise ValueError('备注需要1到4000个字符')
        note_id = identifier('note')
        with self.transaction() as connection:
            self.require_job(connection, job_id)
            if asset_id:
                asset = self.require_asset(connection, asset_id)
                if asset['job_id'] != job_id:
                    raise ValueError('图像不属于该作业')
            connection.execute('INSERT INTO notes VALUES(?,?,?,?,?)', (note_id, job_id, asset_id, body, utc_now()))
            self.touch(connection, job_id)
            self.event(connection, job_id, 'note_added', {'note_id': note_id})
        return {'id': note_id, 'text': body}

    def lineage(self, asset_id):
        chain = []
        seen = set()
        with self.connect() as connection:
            cursor = asset_id
            while cursor:
                if cursor in seen:
                    raise RuntimeError('图像来源链出现循环')
                seen.add(cursor)
                row = self.require_asset(connection, cursor)
                chain.append(self.asset_row(row))
                cursor = row['parent_id']
        return chain

    def events(self, job_id, limit=100):
        bounded = max(1, min(int(limit), 1000))
        with self.connect() as connection:
            self.require_job(connection, job_id)
            rows = connection.execute('SELECT * FROM events WHERE job_id=? ORDER BY sequence DESC LIMIT ?', (job_id, bounded)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item['payload'] = json.loads(item['payload'])
            results.append(item)
        return results

    def health(self):
        with self.connect() as connection:
            jobs = connection.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
            assets = connection.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
            operations = connection.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
            integrity = connection.execute('PRAGMA quick_check').fetchone()[0]
        return {'ok': integrity == 'ok', 'jobs': jobs, 'assets': assets, 'operations': operations, 'database': 'sqlite-wal', 'persistent_root': str(self.root)}

    def export_manifest(self, job_id):
        job = self.get_job(job_id)
        job['events'] = self.events(job_id, 1000)
        job['exported_at'] = utc_now()
        job['format'] = 'printlab-job-v1'
        job['limitations'] = ['RGB数值工作流，不代替ICC印刷认证', 'CMYK模型为教学与小型栅格打样近似', '图像不超过419万像素']
        return job

    def export_archive(self, job_id, include_arrays=False):
        manifest = self.export_manifest(job_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            archive.writestr('manifest.json', self.serialize(manifest))
            for asset in manifest['assets']:
                array = self.load_array(asset['id'])
                if asset['channels'] == 4:
                    from .pixels import tiff_bytes
                    archive.writestr('images/' + asset['id'] + '.tiff', tiff_bytes(array, manifest['dpi']))
                else:
                    archive.writestr('images/' + asset['id'] + '.png', png_bytes(array))
                if include_arrays:
                    with self.connect() as connection:
                        row = self.require_asset(connection, asset['id'])
                        archive.writestr('arrays/' + asset['id'] + '.npy', self.read_blob(row['blob']))
            for operation in manifest['operations']:
                archive.writestr('reports/' + operation['id'] + '.json', self.serialize(operation))
        return stream.getvalue()
