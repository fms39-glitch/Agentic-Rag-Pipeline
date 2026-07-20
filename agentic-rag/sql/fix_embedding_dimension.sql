-- Run this in the Supabase SQL Editor if you already created
-- document_chunks with vector(1024) from the original script.
--
-- Safe to run when the table is empty (ingest failed before inserting rows).
-- If you already have rows, truncate first: truncate table public.document_chunks;

drop index if exists public.document_chunks_embedding_idx;

alter table public.document_chunks
  alter column embedding type vector(384);

create index document_chunks_embedding_idx
  on public.document_chunks
  using hnsw (embedding vector_cosine_ops);

create or replace function public.match_documents(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id uuid,
  content text,
  embedding vector(384),
  source_file text,
  page_number int,
  chunk_index int,
  created_at timestamptz,
  similarity float
)
language sql
stable
as $$
  select
    document_chunks.id,
    document_chunks.content,
    document_chunks.embedding,
    document_chunks.source_file,
    document_chunks.page_number,
    document_chunks.chunk_index,
    document_chunks.created_at,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from public.document_chunks
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;
