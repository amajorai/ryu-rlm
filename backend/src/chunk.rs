//! Structure-aware chunking — turning a pile of documents into the addressable
//! units the root model plans over.
//!
//! The root model never receives the corpus. What it receives is an *outline*: one
//! line per chunk, with the chunk's id, its source document, its line span, and its
//! first heading or first non-blank line. Every operator it can then issue —
//! `peek`, `grep`, `map`, `recurse` — addresses chunks by that id. So chunk
//! boundaries are the app's coordinate system, and two properties matter more than
//! any compression ratio:
//!
//! 1. **A chunk never straddles two documents.** A `map` answer is attributed to a
//!    file; a chunk spanning the seam between two files would produce a citation
//!    that points at both and means neither.
//! 2. **A chunk never splits a line.** Every span reported to a person or a model
//!    is `path:line`, and half a line has no line number that is true.
//!
//! Inside a document the splitter prefers boundaries a human would recognise —
//! Markdown ATX headings first, then blank-line paragraph breaks — and only falls
//! back to a hard line cut when a single run of text exceeds the target on its own.
//! That preference is what makes `outline` readable: a chunk that starts at a
//! heading can label itself with that heading, which is the whole reason the root
//! model can pick chunks intelligently instead of scanning them all.
//!
//! There is deliberately **no overlap** between chunks. Overlap is the usual RAG
//! answer to "a fact landed on a boundary", but it breaks the citation contract:
//! the same sentence would live at two ids, `map` would report it twice, and the
//! fold would double-count it as corroboration. The boundary problem is answered
//! instead by [`Chunk::neighbors`] — an operator can always ask for the chunk
//! before or after — and by `grep`, which searches the flat text and reports which
//! chunk each hit fell in, so a match near a seam is still findable.

use serde::{Deserialize, Serialize};

/// Target chunk size in characters. Chosen so a `map` sub-call — one chunk plus a
/// short instruction — sits comfortably inside even a small local model's window,
/// because the sub-model is meant to be the CHEAP one. Chunks are a soft target,
/// not a cap: a paragraph longer than this is kept whole rather than cut mid-thought.
pub const DEFAULT_TARGET_CHARS: usize = 6_000;

/// Hard ceiling on a single chunk. A run of text with no paragraph break at all —
/// a minified bundle, a one-line CSV, a base64 blob — would otherwise produce one
/// chunk the size of the whole file and defeat the entire mechanism. At this size
/// the splitter cuts on a line boundary regardless of structure.
pub const MAX_CHARS: usize = 24_000;

/// One addressable unit of the context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable index into the context's chunk vector. This is the id every operator
    /// uses and every citation reports, so it must not shift after creation — which
    /// is why a context is immutable once built.
    pub id: usize,
    /// Where this text came from: a file path, a URL, a conversation id, or
    /// `"text"` for a pushed blob. Reported in citations verbatim.
    pub source: String,
    /// 1-based inclusive line span within `source`, so a citation reads `path:120`
    /// and lands on the right line in an editor.
    pub start_line: usize,
    pub end_line: usize,
    /// The chunk's own text.
    pub text: String,
    /// The heading (or first non-blank line) this chunk opens with, trimmed to a
    /// readable length. This is the ONLY per-chunk content the root model sees for
    /// free, so it is what its chunk selection is actually based on.
    pub label: String,
}

impl Chunk {
    pub fn chars(&self) -> usize {
        self.text.chars().count()
    }

    /// The ids immediately before and after this one, bounded by the corpus. The
    /// answer to a fact that landed on a boundary — see the module note on why
    /// there is no overlap.
    pub fn neighbors(&self, total: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if self.id > 0 {
            out.push(self.id - 1);
        }
        if self.id + 1 < total {
            out.push(self.id + 1);
        }
        out
    }
}

/// One source document handed to the chunker.
#[derive(Clone, Debug)]
pub struct Document {
    pub source: String,
    pub text: String,
    /// The line number `text` starts at within `source`, 1-based.
    ///
    /// Almost always 1. It is not 1 when a *sub-context* is built for a `recurse`
    /// step out of chunks taken from the middle of a file: without this the nested
    /// run would re-number those lines from 1, and every citation the recursion
    /// produced would point at the wrong place in the real file while looking
    /// perfectly plausible. Recursion is the app's main mechanism, so a citation
    /// that silently stops being true at depth 1 would poison most real answers.
    pub first_line: usize,
}

impl Document {
    /// A whole document, starting at line 1.
    pub fn new(source: impl Into<String>, text: impl Into<String>) -> Document {
        Document {
            source: source.into(),
            text: text.into(),
            first_line: 1,
        }
    }

    /// An excerpt that begins partway into `source`.
    pub fn excerpt(
        source: impl Into<String>,
        text: impl Into<String>,
        first_line: usize,
    ) -> Document {
        Document {
            source: source.into(),
            text: text.into(),
            first_line: first_line.max(1),
        }
    }
}

/// Split documents into chunks, numbering them consecutively across the whole
/// corpus in document order.
pub fn chunk_documents(docs: &[Document], target: usize) -> Vec<Chunk> {
    let target = target.clamp(500, MAX_CHARS);
    let mut out = Vec::new();
    for doc in docs {
        split_one(doc, target, &mut out);
    }
    out
}

/// Split a single document, appending to `out` so ids stay consecutive across the
/// corpus.
fn split_one(doc: &Document, target: usize, out: &mut Vec<Chunk>) {
    // Work in lines throughout: every boundary decision is a line boundary, and
    // carrying line numbers alongside is what makes citations true.
    let lines: Vec<&str> = doc.text.lines().collect();
    if lines.is_empty() {
        return;
    }

    let base = doc.first_line.max(1);
    let mut buf: Vec<&str> = Vec::new();
    let mut buf_chars = 0usize;
    let mut buf_start = base; // 1-based, offset into the real source

    for (idx, line) in lines.iter().enumerate() {
        let line_no = base + idx;
        let line_chars = line.chars().count() + 1; // + the newline we will re-join with

        // A heading opens a new chunk, but only once the current one has real
        // content — otherwise a document of stacked headings (`# A` / `## B` / …)
        // would emit one chunk per heading and nothing else.
        let heading_break = is_heading(line) && buf_chars >= target / 3;
        // Past target, break at the first paragraph boundary rather than mid-prose.
        let paragraph_break = line.trim().is_empty() && buf_chars >= target;
        // Structure is a PREFERENCE, not a requirement. A source file, a log or a
        // CSV has no headings and no blank lines, so waiting for one means waiting
        // until `MAX_CHARS` — which made `target` silently meaningless for exactly
        // the inputs people point this at most. Past twice the target, cut on the
        // next line boundary and take the loss in tidiness.
        let soft_ceiling = target.saturating_mul(2).min(MAX_CHARS);
        let forced = buf_chars + line_chars > soft_ceiling || buf_chars + line_chars > MAX_CHARS;

        if (heading_break || paragraph_break || forced) && !buf.is_empty() {
            push_chunk(doc, &buf, buf_start, line_no - 1, out);
            buf.clear();
            buf_chars = 0;
            buf_start = line_no;
        }

        buf.push(line);
        buf_chars += line_chars;
    }

    if !buf.is_empty() {
        push_chunk(doc, &buf, buf_start, base + lines.len() - 1, out);
    }
}

fn push_chunk(
    doc: &Document,
    lines: &[&str],
    start_line: usize,
    end_line: usize,
    out: &mut Vec<Chunk>,
) {
    let text = lines.join("\n");
    // A chunk of pure whitespace carries no information but would still occupy an
    // outline row and a `map` call, so it is dropped rather than numbered.
    if text.trim().is_empty() {
        return;
    }
    let label = label_for(lines);
    out.push(Chunk {
        id: out.len(),
        source: doc.source.clone(),
        start_line,
        end_line,
        text,
        label,
    });
}

/// Markdown ATX heading (`#` … `######`). Deliberately NOT setext (`===` under a
/// line): detecting that needs lookahead, and a false positive would split a table
/// of dashes into one chunk per row.
fn is_heading(line: &str) -> bool {
    let t = line.trim_start();
    let hashes = t.chars().take_while(|c| *c == '#').count();
    (1..=6).contains(&hashes) && t.chars().nth(hashes).is_some_and(|c| c == ' ')
}

/// The one line that stands in for the chunk in the outline: its heading if it
/// opens with one, else its first non-blank line.
fn label_for(lines: &[&str]) -> String {
    let raw = lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .copied()
        .unwrap_or("");
    let cleaned = raw.trim().trim_start_matches('#').trim();
    let mut s: String = cleaned.chars().take(96).collect();
    if cleaned.chars().count() > 96 {
        s.push('…');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(source: &str, text: &str) -> Document {
        Document::new(source, text)
    }

    #[test]
    fn an_excerpt_numbers_its_lines_from_where_it_really_starts() {
        // The recursion citation guard: a sub-context built from the middle of a
        // file must report the file's real line numbers, not 1-based ones.
        let chunks = chunk_documents(&[Document::excerpt("f.rs", "a\nb\nc", 400)], 6_000);
        assert_eq!(chunks[0].start_line, 400);
        assert_eq!(chunks[0].end_line, 402);
    }

    #[test]
    fn a_chunk_never_straddles_two_documents() {
        // Both documents are far under the target, so a size-driven splitter would
        // happily pack them together — and the resulting citation would name one
        // file for text that came from two.
        let chunks = chunk_documents(&[doc("a.md", "alpha"), doc("b.md", "beta")], 6_000);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].source, "a.md");
        assert_eq!(chunks[1].source, "b.md");
        assert_eq!(chunks[1].id, 1, "ids run consecutively across the corpus");
    }

    #[test]
    fn line_spans_are_true_and_one_based() {
        let text = (1..=200)
            .map(|i| format!("line {i} {}", "x".repeat(60)))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_documents(&[doc("big.txt", &text)], 1_000);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 1);
        for pair in chunks.windows(2) {
            assert_eq!(
                pair[1].start_line,
                pair[0].end_line + 1,
                "spans must tile the file with no gap and no overlap"
            );
        }
        assert_eq!(chunks.last().unwrap().end_line, 200);
    }

    #[test]
    fn headings_open_new_chunks_once_there_is_content_to_close() {
        let text = "# One\n".to_owned()
            + &"body line\n".repeat(40)
            + "# Two\n"
            + &"other body\n".repeat(40);
        let chunks = chunk_documents(&[doc("d.md", &text)], 300);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].label, "One");
        assert!(
            chunks.iter().any(|c| c.label == "Two"),
            "the second heading must start a chunk, not be buried mid-chunk"
        );
    }

    #[test]
    fn stacked_headings_do_not_produce_one_chunk_each() {
        // The regression the `buf_chars >= target / 3` guard exists for: a document
        // that is nothing but headings would otherwise emit an outline row per line.
        let text = "# A\n## B\n### C\n#### D\n";
        let chunks = chunk_documents(&[doc("toc.md", text)], 6_000);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn a_document_with_no_blank_lines_still_honours_the_target() {
        // The regression that made `target` meaningless for source files, logs and
        // CSVs: with no heading and no blank line to break on, everything stayed in
        // one chunk until the 24k hard ceiling.
        let text = (1..=200)
            .map(|i| format!("const x{i} = compute({i});"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_documents(&[doc("a.ts", &text)], 1_000);
        assert!(
            chunks.len() > 1,
            "expected several chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(c.chars() <= 2_000, "chunk {} is {} chars", c.id, c.chars());
        }
    }

    #[test]
    fn a_run_with_no_structure_is_still_cut_at_the_hard_ceiling() {
        // One "paragraph" of 40k characters across many lines, no blank lines and no
        // headings — the minified-bundle shape. Without the forced cut this is one
        // chunk the size of the file, and the whole mechanism degrades to "put the
        // corpus in the prompt".
        let text = "x".repeat(400) + "\n";
        let text = text.repeat(100); // 100 lines × ~401 chars ≈ 40k
        let chunks = chunk_documents(&[doc("bundle.js", &text)], 6_000);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(
                c.chars() <= MAX_CHARS,
                "chunk {} exceeded the ceiling",
                c.id
            );
        }
    }

    #[test]
    fn whitespace_only_regions_are_not_numbered() {
        let chunks = chunk_documents(&[doc("blank.txt", "\n\n\n\n")], 6_000);
        assert!(chunks.is_empty());
    }

    #[test]
    fn neighbors_are_bounded_by_the_corpus() {
        let chunks = chunk_documents(&[doc("a.md", "one"), doc("b.md", "two")], 6_000);
        assert_eq!(chunks[0].neighbors(2), vec![1]);
        assert_eq!(chunks[1].neighbors(2), vec![0]);
    }
}
