package types

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"github.com/yanyiwu/gojieba"
)

// Jieba is a global instance of Chinese text segmentation tool
var Jieba *gojieba.Jieba = newJieba()

func newJieba() (jieba *gojieba.Jieba) {
	defer func() {
		if err := recover(); err != nil {
			log.Printf("WARNING: jieba initialization failed, using fallback tokenization: %v", err)
			jieba = nil
		}
	}()

	dictDir := os.Getenv("JIEBA_DICT_DIR")
	if dictDir == "" {
		return gojieba.NewJieba()
	}

	return gojieba.NewJieba(
		filepath.Join(dictDir, "jieba.dict.utf8"),
		filepath.Join(dictDir, "hmm_model.utf8"),
		filepath.Join(dictDir, "user.dict.utf8"),
		filepath.Join(dictDir, "idf.utf8"),
		filepath.Join(dictDir, "stop_words.utf8"),
	)
}

func fallbackTokenize(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if !containsChinese(text) {
		return strings.Fields(text)
	}

	var (
		tokens     []string
		segment    []rune
		segChinese bool
		segInit    bool
	)

	flush := func() {
		if len(segment) == 0 {
			return
		}
		part := string(segment)
		if segChinese {
			runes := []rune(part)
			if len(runes) <= 2 {
				tokens = append(tokens, part)
			} else {
				// Use overlapping 2-rune n-grams as a lightweight approximation for Chinese tokenization.
				for i := 0; i < len(runes)-1; i++ {
					tokens = append(tokens, string(runes[i:i+2]))
				}
			}
		} else {
			tokens = append(tokens, strings.Fields(part)...)
		}
		segment = segment[:0]
	}

	for _, r := range text {
		isChinese := unicode.Is(unicode.Han, r)
		if !segInit {
			segChinese = isChinese
			segInit = true
		}
		if isChinese != segChinese {
			flush()
			segChinese = isChinese
		}
		segment = append(segment, r)
	}
	flush()
	return tokens
}

func containsChinese(text string) bool {
	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

// JiebaCut safely tokenizes text with jieba and falls back when dictionaries are unavailable.
// The hmm parameter controls Hidden Markov Model-based segmentation for unknown words.
// In fallback mode, hmm is ignored and simplified tokenization is used.
func JiebaCut(text string, hmm bool) []string {
	if Jieba == nil {
		return fallbackTokenize(text)
	}
	return Jieba.Cut(text, hmm)
}

// JiebaCutForSearch safely tokenizes text with jieba search mode and falls back when unavailable.
// The hmm parameter controls Hidden Markov Model-based segmentation for unknown words.
// Search mode returns finer-grained tokens than JiebaCut for better recall in retrieval.
// In fallback mode, hmm is ignored and simplified tokenization is used.
func JiebaCutForSearch(text string, hmm bool) []string {
	if Jieba == nil {
		return fallbackTokenize(text)
	}
	return Jieba.CutForSearch(text, hmm)
}

// EvaluationStatue represents the status of an evaluation task
type EvaluationStatue int

const (
	EvaluationStatuePending EvaluationStatue = iota // Task is waiting to start
	EvaluationStatueRunning                         // Task is in progress
	EvaluationStatueSuccess                         // Task completed successfully
	EvaluationStatueFailed                          // Task failed
)

// EvaluationTask contains information about an evaluation task
type EvaluationTask struct {
	ID        string `json:"id"`         // Unique task ID
	TenantID  uint64 `json:"tenant_id"`  // Tenant/Organization ID
	DatasetID string `json:"dataset_id"` // Dataset ID for evaluation

	StartTime time.Time        `json:"start_time"`        // Task start time
	Status    EvaluationStatue `json:"status"`            // Current task status
	ErrMsg    string           `json:"err_msg,omitempty"` // Error message if failed

	Total    int `json:"total,omitempty"`    // Total items to evaluate
	Finished int `json:"finished,omitempty"` // Completed items count
}

// EvaluationDetail contains detailed evaluation information
type EvaluationDetail struct {
	Task   *EvaluationTask `json:"task"`             // Evaluation task info
	Params *ChatManage     `json:"params"`           // Evaluation parameters
	Metric *MetricResult   `json:"metric,omitempty"` // Evaluation metrics
}

// String returns JSON representation of EvaluationTask
func (e *EvaluationTask) String() string {
	b, _ := json.Marshal(e)
	return string(b)
}

// MetricInput contains input data for metric calculation
type MetricInput struct {
	RetrievalGT  [][]int // Ground truth for retrieval
	RetrievalIDs []int   // Retrieved IDs

	GeneratedTexts string // Generated text for evaluation
	GeneratedGT    string // Ground truth text for comparison
}

// MetricResult contains evaluation metrics
type MetricResult struct {
	RetrievalMetrics  RetrievalMetrics  `json:"retrieval_metrics"`  // Retrieval performance metrics
	GenerationMetrics GenerationMetrics `json:"generation_metrics"` // Text generation quality metrics
}

// RetrievalMetrics contains metrics for retrieval evaluation
type RetrievalMetrics struct {
	Precision float64 `json:"precision"` // Precision score
	Recall    float64 `json:"recall"`    // Recall score

	NDCG3  float64 `json:"ndcg3"`  // Normalized Discounted Cumulative Gain at 3
	NDCG10 float64 `json:"ndcg10"` // Normalized Discounted Cumulative Gain at 10
	MRR    float64 `json:"mrr"`    // Mean Reciprocal Rank
	MAP    float64 `json:"map"`    // Mean Average Precision
}

// GenerationMetrics contains metrics for text generation evaluation
type GenerationMetrics struct {
	BLEU1 float64 `json:"bleu1"` // BLEU-1 score
	BLEU2 float64 `json:"bleu2"` // BLEU-2 score
	BLEU4 float64 `json:"bleu4"` // BLEU-4 score

	ROUGE1 float64 `json:"rouge1"` // ROUGE-1 score
	ROUGE2 float64 `json:"rouge2"` // ROUGE-2 score
	ROUGEL float64 `json:"rougel"` // ROUGE-L score
}

// EvalState represents different stages of evaluation process
type EvalState int

const (
	StateBegin             EvalState = iota // Evaluation started
	StateAfterQaPairs                       // After loading QA pairs
	StateAfterDataset                       // After processing dataset
	StateAfterEmbedding                     // After generating embeddings
	StateAfterVectorSearch                  // After vector search
	StateAfterRerank                        // After reranking
	StateAfterComplete                      // After completion
	StateEnd                                // Evaluation ended
)
