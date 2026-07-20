-- Run this script in the Supabase SQL Editor.
-- Creates the document_chunks table and a cosine-similarity search function.

create extension if not exists vector;

create table if not exists public.document_chunks (
    id uuid primary key default gen_random_uuid (),
    content text not null,
    embedding vector (384) not null,
    source_file text not null,
    page_number int not null,
    chunk_index int not null,
    created_at timestamptz not null default now()
);

-- Speeds up nearest-neighbor search ordered by cosine distance (<=>).
create index if not exists document_chunks_embedding_idx on public.document_chunks using hnsw (embedding vector_cosine_ops);

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