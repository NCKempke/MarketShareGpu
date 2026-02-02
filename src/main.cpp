#include "argparse.hpp"

#include "markshare.hpp"
#include "pairs_tuple.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits> // For CHAR_BIT
#include <cstddef>
#include <execution>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <limits>
#include <queue>
#include <utility>
#include <string>
#include <unordered_map>

#include <omp.h>

#ifdef WITH_GPU
#include "cuda_kernels.cuh"
#endif

/* 3000000000 ~= 56 GB of active storage requirement. 4000000000 goes OOM on H200. 3500000000 works and goes up to 63.6 GB. 3900000000 also works and is about ~= 70.11 */
size_t max_pairs_per_chunk = 3500000000;

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&...args)
{
    TimeVar t1 = timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow() - t1);
}

template <typename T>
void print_vector(const std::vector<T> vec)
{
    std::cout << "[";
    for (auto &e : vec)
        std::cout << " " << e;
    std::cout << "]\n";
}

size_t highestSetBit(size_t value)
{
    if (value == 0)
        return -1; // No bits are set
#if defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 8
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clzll(value);
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 4
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clz(value);
#else
#error Unsupported size_t size
#endif
}

size_t countSetBits(size_t num)
{
    return __builtin_popcount(num);
}

void print_bits(size_t value)
{
    if (value == 0)
    {
        std::cout << "0"; // Special case: value is 0
        return;
    }

    // Determine the position of the highest set bit
    size_t msb = 0;
    for (size_t i = sizeof(size_t) * 8; i > 0; --i)
    {
        if (value & (1ULL << (i - 1)))
        {
            msb = i - 1;
            break;
        }
    }

    // Print bits from the highest set bit down to 0
    for (size_t i = msb + 1; i > 0; --i)
    {
        std::cout << ((value & (1ULL << (i - 1))) ? '1' : '0');
    }
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T> &vec,
    const std::vector<std::size_t> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](std::size_t i)
                   { return vec[i]; });
    return sorted_vec;
}

template <typename T>
std::pair<std::vector<T>, std::vector<std::vector<size_t>>> generate_subsets(const std::vector<T> &weights)
{
    size_t n = weights.size();
    size_t total_subsets = 1ULL << n; /* Total subsets is 2^n. */

    printf("Generating %ld possible subsets for as set of size %ld.\n", total_subsets, n);
    std::vector<T> set_weights(total_subsets, 0);
    std::vector<std::vector<size_t>> sets(total_subsets);

    for (size_t pass = 0; pass < n; ++pass)
    {
        const T weight = weights[pass];
        /* Step size corresponds to 2^pass (position of the bit). */
        size_t step = 1ULL << pass;

#pragma omp parallel for
        for (size_t i = 0; i < total_subsets; i += step * 2)
        {
            /* Add `number` to all subsets where the `pass`-th bit is set. */
            for (size_t j = 0; j < step; ++j)
            {
                set_weights[i + step + j] += weight;
                sets[i + step + j].push_back(pass);
            }
        }
    }

    return {set_weights, sets};
}

// Function to sort an array and obtain sorted indices
template <typename T>
std::vector<size_t> sort_indices(const std::vector<T> &arr, bool ascending)
{
    size_t n = arr.size();

    // Create indices list from 0 to n-1 using std::iota
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on corresponding values in the array
    if (ascending)
    {
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] < arr[i2]; });
    }
    else
    {
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] > arr[i2]; });
    }

    return indices;
}

size_t print_subset_and_compute_sum(const std::vector<size_t> &numbers, size_t index)
{
    std::cout << "Subset for index " << index << " (binary " << std::bitset<64>(index) << "): ";
    size_t sum = 0;

    bool hasElements = false;
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
        {
            if (hasElements)
            {
                std::cout << ", ";
            }
            std::cout << numbers[i];
            sum += numbers[i];
            hasElements = true;
        }
    }

    if (!hasElements)
    {
        std::cout << "Empty";
    }

    std::cout << std::endl;
    return sum;
}

std::vector<size_t> extract_subset(const std::vector<size_t> &numbers, size_t index)
{
    std::vector<size_t> indices;

    size_t position = 0;
    while (index > 0)
    {
        if (index & 1)
            indices.push_back(numbers[position]);
        index >>= 1;
        ++position;
    }
    return indices;
}

void concat_vectors(std::vector<size_t> &concat_vec, size_t &concat_len, const std::vector<const std::vector<size_t> *> &vectors, const std::vector<size_t> offsets)
{
    assert(vectors.size() == offsets.size());

    concat_len = 0;
    for (const auto &vec : vectors)
    {
        concat_len += vec->size();
    }

    assert(concat_len <= concat_vec.size());

    size_t pos = 0;
    for (size_t ivec = 0; ivec < vectors.size(); ++ivec)
    {
        const auto &vec = *vectors[ivec];
        const auto offset = offsets[ivec];

        for (size_t num : vec)
        {
            concat_vec[pos] = (num + offset);
            ++pos;
        }
    }

    assert(pos == concat_len);
}

void print_four_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);
    auto sum3 = print_subset_and_compute_sum(list3, index_list3);
    auto sum4 = print_subset_and_compute_sum(list4, index_list4);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " + " << sum3 << " + " << sum4 << " = " << sum1 + sum2 + sum3 + sum4 << std::endl;
}

void append_solution_to_file(std::ofstream &sol_file, const std::vector<size_t> &numbers, size_t index)
{
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
            sol_file << 1;
        else
            sol_file << 0;
    }
}

void write_four_list_solution_to_file(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4, const std::string &instance_name)
{
    const std::string sol_name = instance_name + ".sol";
    printf("Writing solution to %s\n", sol_name.c_str());
    std::ofstream sol_file(sol_name);

    append_solution_to_file(sol_file, list1, index_list1);
    append_solution_to_file(sol_file, list2, index_list2);
    append_solution_to_file(sol_file, list3, index_list3);
    append_solution_to_file(sol_file, list4, index_list4);

    sol_file << std::endl;
}

size_t custom_hash_cpu(size_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

template <const bool ENCODE_REQUIRED>
std::vector<size_t> flatten_and_encode_tuples_cpu(const std::vector<PairsTuple>& tuples, size_t n_tuples, const std::vector<size_t> &scores1, const std::vector<size_t> &scores2, const MarkShareFeas &ms_inst, size_t row_offset)
{
    const size_t m_rows_left = ms_inst.m() - row_offset;
    std::vector<size_t> hashes (n_tuples);

#pragma omp parallel for
    for (size_t i_tuple = 0; i_tuple < tuples.size(); ++i_tuple)
    {
        auto first = tuples[i_tuple].pairs_first;
        auto pair_second_beg = tuples[i_tuple].pairs_second_beg;
        auto pair_second_end = pair_second_beg + tuples[i_tuple].pairs_n_second;
        auto pair_offset = tuples[i_tuple].pairs_offset;

        for (size_t second = pair_second_beg; second < pair_second_end; ++second)
        {
            size_t key = 0;

            /* Compute the hash of this tuple. */
            for (size_t i_row = 0; i_row < m_rows_left; ++i_row) {
                /* Compute the pair's score of this row and add it (encoded) to key. */
                size_t row_score = scores1[first * m_rows_left + i_row] + scores2[second * m_rows_left + i_row];

                if (ENCODE_REQUIRED)
                    row_score = ms_inst.b()[i_row + row_offset] - row_score;

                key ^= custom_hash_cpu(row_score) + 0x9e3779b9 + (key << 6) + (key >> 2);
            }

            hashes[pair_offset] = key;
            ++pair_offset;
        }
    }

    return hashes;
}

void compute_scores_cpu(const MarkShareFeas &ms_inst, std::vector<size_t> &scores, const std::vector<std::vector<size_t>> &subsets, size_t col_offset, size_t row_offset)
{
    const size_t m_rows_left = ms_inst.m() - row_offset;
    assert(ms_inst.m() >= row_offset);

#pragma omp parallel for
    for (size_t i = 0; i < subsets.size(); ++i)
    {
        ms_inst.compute_value(subsets[i], col_offset, row_offset, scores.data() + i * m_rows_left);
    }
}

/* For each tuple consisting of one element in set1_scores and a range of elements in set2_scores, compute its partial right hand side. */
void combine_scores_cpu(const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, size_t m_rows, const std::vector<PairsTuple> &tuples, std::vector<size_t> &scores_pairs, size_t row_offset)
{
    const size_t m_rows_left = m_rows - row_offset;

#pragma omp parallel for
    for (size_t i_tuple = 0; i_tuple < tuples.size(); ++i_tuple)
    {
        auto first = tuples[i_tuple].pairs_first;
        auto pair_second_beg = tuples[i_tuple].pairs_second_beg;
        auto pair_second_end = pair_second_beg + tuples[i_tuple].pairs_n_second;
        auto pair_offset = tuples[i_tuple].pairs_offset;

        for (size_t second = pair_second_beg; second < pair_second_end; ++second)
        {
            for (size_t i_row = 0; i_row < m_rows_left; ++i_row)
                scores_pairs[pair_offset * m_rows_left + i_row] = set1_scores[first * m_rows_left + i_row] + set2_scores[second * m_rows_left + i_row];
            ++pair_offset;
        }
    }
}

void print_info_line(
#ifdef WITH_GPU
    const GpuData &gpu_data,
    bool run_on_gpu,
#endif
    size_t i_iter, double time, size_t score1, size_t score2, size_t n_q1, size_t n_q2)
{
    bool print = false;
    if (i_iter < 10)
        print = true;
    else if (i_iter < 100 && i_iter % 10 == 0)
        print = true;
    else if (i_iter < 10000 && i_iter % 100 == 0)
        print = true;
    else if (i_iter % 1000 == 0)
        print = true;

    if (print)
    {
#ifdef WITH_GPU
        if (run_on_gpu)
        {
            const double n_gb = gpu_data.get_gb_allocated();
            printf("%5ld %8.2fs [%.6f GB]: %6ld + %6ld; %ld x %ld possible solutions\n", i_iter, time, n_gb, score1, score2, n_q1, n_q2);
        }
        else
#endif
        {
            printf("%5ld %8.2fs: %6ld + %6ld; %ld x %ld possible solutions\n", i_iter, time, score1, score2, n_q1, n_q2);
        }
    }
}

bool verify_solution(const std::pair<size_t, size_t> &solution, const PairsTuple *same_score_q1, size_t n_tuples_q1, const PairsTuple *same_score_q2, size_t n_tuples_q2, const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, const MarkShareFeas &ms_inst)
{
    /* Print and verify the solution! */
    /* Get the correct pairs. */
    size_t pos_q1 = 0;
    size_t pos_q2 = 0;
    (void)n_tuples_q1;
    (void)n_tuples_q2;

    while (solution.first >= same_score_q1[pos_q1].pairs_offset + same_score_q1[pos_q1].pairs_n_second)
    {
        ++pos_q1;
        assert(pos_q1 < n_tuples_q1);
    }
    while (solution.second >= same_score_q2[pos_q2].pairs_offset + same_score_q2[pos_q2].pairs_n_second)
    {
        ++pos_q2;
        assert(pos_q2 < n_tuples_q2);
    }

    assert(solution.first - same_score_q1[pos_q1].pairs_offset < same_score_q1[pos_q1].pairs_n_second);
    assert(solution.second - same_score_q2[pos_q2].pairs_offset < same_score_q2[pos_q2].pairs_n_second);

    const size_t pair_q1_second = same_score_q1[pos_q1].pairs_second_beg + solution.first - same_score_q1[pos_q1].pairs_offset;
    const size_t pair_q2_second = same_score_q2[pos_q2].pairs_second_beg + solution.second - same_score_q2[pos_q2].pairs_offset;
    std::pair<size_t, size_t> pair_q1 = {same_score_q1[pos_q1].pairs_first, pair_q1_second};
    std::pair<size_t, size_t> pair_q2 = {same_score_q2[pos_q2].pairs_first, pair_q2_second};

    std::vector<size_t> solution_1d(subset_sum_1d.size());

    const std::vector<const std::vector<size_t> *> vectors = {&set1_subsets[pair_q1.first], &set2_subsets_sorted_asc[pair_q1.second], &set3_subsets[pair_q2.first], &set4_subsets_sorted_desc[pair_q2.second]};

    size_t len;
    concat_vectors(solution_1d, len, vectors, offsets);

    /* We found a solution. Construct it, print it, and return. */
    if (!ms_inst.is_solution_feasible(solution_1d, len))
    {
#ifndef NDEBUG
        printf("Error, solution is not feasible!\n");
#endif
        return false;
    }

    return true;
}

template <typename T>
std::pair<size_t, T> max_encodable_dimension(size_t max_coeff, size_t n_cols)
{
    constexpr T max_index = std::numeric_limits<T>::max();

    static_assert(std::numeric_limits<T>::max() >= std::numeric_limits<size_t>::max());
    assert(max_coeff > 1);

    /* TODO: we need to properly check for an overflow here. */
    const T basis = static_cast<T>(n_cols) * static_cast<T>(max_coeff) + 1;

    /* Given a vector (x1, x2, .. ) we reduce dimensions as x1 * max_sum^0 + x2 * max_sum^1 + x3 * max_sum^2 ...
     * When encoding into T we need to guarantee that the highest dimension, d, still fits into our index type. Define the basis B as
     *   B := (max_sum + 1)
     * Then
     *   sum_0^d-1 (B - 1) (max_sum + 1)^k <= max_index
     * <=> geometric sum
     *   (B - 1) [B^d - 1] / [B - 1] = (B - 1) * [B^d - 1] / (B - 1) <= max_index
     * <=>
     *   B^d <= max_index + 1
     * <=>
     *   d_max = |_ log_(max_sum + 1) (max_index + 1) _|
     */
    size_t max_dim = static_cast<size_t>(std::floor(std::log(max_index) / std::log(basis)));

    printf("Max reducible dimension is %ld (encoded with basis %ld)\n", max_dim, static_cast<size_t>(basis));
    return {max_dim, basis};
}

template <bool ascending, typename T, typename F>
void extract_pairs_from_heap(std::vector<PairsTuple> &pairs_same_score, std::vector<std::pair<size_t, size_t>> &heap, size_t score_pair, std::vector<size_t> &chunks_beg, std::vector<size_t> &chunks_n_pairs, size_t &n_pairs_total, const std::vector<T> &first_weights, const std::vector<T> &second_weights, const std::vector<T> &second_weights_sorted_asc, F &&cmp)
{
    /* Counter for pairs stored in current chunk. */
    size_t n_pairs_chunk = 0;

    /* For each element a in the heap with score(a) == score_pair, collect all solutions. */
    while (!heap.empty() && first_weights[heap.front().first] + second_weights_sorted_asc[heap.front().second] == score_pair)
    {
        const auto pair1_same_score = heap.front();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.pop_back();

        const size_t pos_second_weights_beg = pair1_same_score.second;
        size_t pos_second_weights_end = pos_second_weights_beg;

        /* Iterate the second elements. */
        const auto pos2_val = second_weights_sorted_asc[pos_second_weights_end];

        while (pos_second_weights_end < second_weights.size() && pos2_val == second_weights_sorted_asc[pos_second_weights_end])
            ++pos_second_weights_end;

        size_t n_pairs = pos_second_weights_end - pos_second_weights_beg;
        assert(n_pairs <= max_pairs_per_chunk);

        if (n_pairs + n_pairs_chunk >= max_pairs_per_chunk)
        {
            chunks_beg.push_back(pairs_same_score.size());
            chunks_n_pairs.push_back(n_pairs_chunk);
            n_pairs_chunk = 0;
        }

        pairs_same_score.emplace_back(pair1_same_score.first, pos_second_weights_beg, n_pairs, n_pairs_chunk);
        n_pairs_chunk += n_pairs;
        n_pairs_total += n_pairs;

        if (pos_second_weights_end < second_weights.size())
        {
            if (ascending)
                assert(score_pair < first_weights[pair1_same_score.first] + second_weights_sorted_asc[pos_second_weights_end]);
            else
                assert(score_pair > first_weights[pair1_same_score.first] + second_weights_sorted_asc[pos_second_weights_end]);

            heap.emplace_back(pair1_same_score.first, pos_second_weights_end);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    chunks_n_pairs.push_back(n_pairs_chunk);
    chunks_beg.push_back(pairs_same_score.size());
}

std::vector<size_t> find_equal_hashes_cpu(std::vector<size_t>& hashes_required, const std::vector<size_t>& hashes_search, bool sort_required = true)
{
    /* The shorter array will be encoded as required and will be sorted. */
    const size_t n_search = hashes_search.size();

    /* Compute hashes of required vectors. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval CPU: sort required     ");
    if (sort_required)
    {
        std::sort(std::execution::par_unseq, hashes_required.begin(), hashes_required.end());
    }

    profiler = std::make_unique<ScopedProfiler>("Eval CPU: binary search     ");

    std::vector<bool> result(n_search);

    /* Parallel binary search for each hash. */
#pragma omp parallel for
    for (size_t i = 0; i < n_search; i++)
    {
        result[i] = std::binary_search(hashes_required.begin(), hashes_required.end(), hashes_search[i]);
    }

    profiler = std::make_unique<ScopedProfiler>("Eval CPU: check results     ");

    std::vector<size_t> hashes;
    for (size_t i = 0; i < n_search; i++)
    {
        if (result[i])
        {
            hashes.push_back(hashes_search[i]);
        }
    }

    profiler.reset();
    return hashes;
}

std::vector<std::pair<size_t, size_t>> find_hash_positions_cpu(const std::vector<size_t>& hashes_required,
                                                                const std::vector<size_t>& hashes_search,
                                                                const std::vector<size_t>& matching_hashes,
                                                                bool encode_first_as_required)
{
    encode_first_as_required = encode_first_as_required || (hashes_required.size() < hashes_search.size());

    std::vector<std::pair<size_t, size_t>> solution_candidates;
    solution_candidates.reserve(matching_hashes.size());

    for (const auto hash : matching_hashes)
    {
        // Find position in required vector
        auto iter_req = std::find(hashes_required.begin(), hashes_required.end(), hash);
        // Find position in search vector
        auto iter_search = std::find(hashes_search.begin(), hashes_search.end(), hash);

        // Assertions (could be removed or replaced with error handling)
        assert(iter_req != hashes_required.end());
        assert(iter_search != hashes_search.end());

        auto pos_req = std::distance(hashes_required.begin(), iter_req);
        auto pos_search = std::distance(hashes_search.begin(), iter_search);

        if (encode_first_as_required)
            solution_candidates.emplace_back(pos_req, pos_search);
        else
            solution_candidates.emplace_back(pos_search, pos_req);
    }

    return solution_candidates;
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_cpu(const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores_sorted_asc, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores_sorted_desc, const std::vector<PairsTuple> &same_score_q1, size_t n_pairs_q1, const std::vector<PairsTuple> &same_score_q2, size_t n_pairs_q2, const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc,
                                                                         const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets)
{
    /* Encode the 2 sets as 'required'. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval CPU: combine + encode  ");
    auto required = flatten_and_encode_tuples_cpu<true>(same_score_q1, n_pairs_q1, set1_scores, set2_scores_sorted_asc, ms_inst, reduce_dim);
    const auto search = flatten_and_encode_tuples_cpu<false>(same_score_q2, n_pairs_q2, set3_scores, set4_scores_sorted_desc, ms_inst, reduce_dim);
    profiler.reset();

    auto hashes = find_equal_hashes_cpu(required, search);

    if (!hashes.empty())
    {
        required = flatten_and_encode_tuples_cpu<true>(same_score_q1, n_pairs_q1, set1_scores, set2_scores_sorted_asc, ms_inst, reduce_dim);
        const std::vector<std::pair<size_t, size_t>> candidates = find_hash_positions_cpu(required, search, hashes, true);

        /* Check all potential solutions. */
        for (const auto &solution_cand : candidates)
        {
            /* Offset each solution candidate by its chunk. */
            const auto feasible = verify_solution(solution_cand, same_score_q1.data(), n_pairs_q1, same_score_q2.data(), n_pairs_q2, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, ms_inst);

            if (feasible)
                return {true, solution_cand};
        }
    }

    return {false, {0, 0}};
}

#ifdef WITH_GPU
std::tuple<bool, size_t, size_t, std::pair<size_t, size_t>> evaluate_gpu(GpuData &gpu_data, const std::vector<PairsTuple> &same_score_q1, const std::vector<PairsTuple> &same_score_q2, const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<size_t> &chunks_q1_beg, const std::vector<size_t> &chunks_q1_n_pairs, const std::vector<size_t> &chunks_q2_beg, const std::vector<size_t> &chunks_q2_n_pairs, size_t n_q1_chunks, size_t n_q2_chunks, const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc,
                                                                         const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets
#ifndef NDEBUG
                                                                         ,
                                                                         size_t n_pairs_q1, size_t n_pairs_q2
#endif
)
{
    assert(n_q1_chunks > 0);
    assert(n_q2_chunks > 0);
    assert(chunks_q1_beg.size() == n_q1_chunks + 1);
    assert(chunks_q2_beg.size() == n_q2_chunks + 1);

    /* Do all this per chunk and quadratically. */
    size_t n_q1_pairs_offset = 0;

    for (size_t i_q1_chunk = 0; i_q1_chunk < n_q1_chunks; ++i_q1_chunk)
    {
        const size_t q1_chunk_beg = chunks_q1_beg[i_q1_chunk];
        const size_t n_pairs_q1_chunk = chunks_q1_n_pairs[i_q1_chunk];
        const size_t n_tuples_q1_chunk = chunks_q1_beg[i_q1_chunk + 1] - q1_chunk_beg;
        assert(q1_chunk_beg + n_tuples_q1_chunk <= same_score_q1.size());

        const PairsTuple *q1_chunk = same_score_q1.data() + q1_chunk_beg;
        size_t n_q2_pairs_offset = 0;

        combine_and_encode_tuples_required_gpu(gpu_data, q1_chunk, n_tuples_q1_chunk, n_pairs_q1_chunk, gpu_data.set1_scores, gpu_data.set2_scores, reduce_dim);
        sort_required_gpu(gpu_data);

        for (size_t i_q2_chunk = 0; i_q2_chunk < n_q2_chunks; ++i_q2_chunk)
        {
            const size_t q2_chunk_beg = chunks_q2_beg[i_q2_chunk];
            const size_t n_pairs_q2_chunk = chunks_q2_n_pairs[i_q2_chunk];
            const size_t n_tuples_q2_chunk = chunks_q2_beg[i_q2_chunk + 1] - q2_chunk_beg;
            assert(q2_chunk_beg + n_tuples_q2_chunk <= same_score_q2.size());

            const PairsTuple *q2_chunk = same_score_q2.data() + q2_chunk_beg;

            combine_and_encode_tuples_search_gpu(gpu_data, q2_chunk, n_tuples_q2_chunk, n_pairs_q2_chunk, gpu_data.set3_scores, gpu_data.set4_scores, reduce_dim);

            const std::vector<size_t> hashes = find_equal_hashes(gpu_data, false);

            if (!hashes.empty())
            {
                /* Retrieve the actual solution. We have to copy encode our arrays once more and look for the hash afterwards. */
                combine_and_encode_tuples_required_gpu(gpu_data, q1_chunk, n_tuples_q1_chunk, n_pairs_q1_chunk, gpu_data.set1_scores, gpu_data.set2_scores, reduce_dim);
                combine_and_encode_tuples_search_gpu(gpu_data, q2_chunk, n_tuples_q2_chunk, n_pairs_q2_chunk, gpu_data.set3_scores, gpu_data.set4_scores, reduce_dim);

                const std::vector<std::pair<size_t, size_t>> candidates = find_hash_positions_gpu(gpu_data, hashes, n_pairs_q1_chunk, n_pairs_q2_chunk, true);

                /* Check all potential solutions. */
                for (const auto &solution_cand : candidates)
                {
                    /* Offset each solution candidate by its chunk. */
                    const auto feasible = verify_solution(solution_cand, q1_chunk, n_tuples_q1_chunk, q2_chunk, n_tuples_q2_chunk, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, ms_inst);

                    if (feasible)
                        return {true, i_q1_chunk, i_q2_chunk, solution_cand};
                }
            }

            n_q2_pairs_offset += n_pairs_q2_chunk;
        }

        assert(n_q2_pairs_offset == n_pairs_q2);
        n_q1_pairs_offset += n_pairs_q1_chunk;
    }

    assert(n_q1_pairs_offset == n_pairs_q1);

    return {false, 0, 0, {0, 0}};
}
#endif

enum BufferState
{
    EMPTY,
    EXTRACTING,
    READY_FOR_EVAL,
    EVALUATING,
    EVALUATED
};

struct PipelineBuffer
{
    std::vector<PairsTuple> same_score_q1, same_score_q2;
    std::vector<size_t> chunks_q1_n_pairs, chunks_q1_beg;
    std::vector<size_t> chunks_q2_n_pairs, chunks_q2_beg;
    size_t n_pairs_q1 = 0, n_pairs_q2 = 0;
    BufferState state = EMPTY;
};

struct EvalResult
{
    bool found = false;
    std::pair<size_t, size_t> solution;
    size_t i_q1_chunk = 0;
    size_t i_q2_chunk = 0;
};

EvalResult evaluate_gpu_or_cpu(PipelineBuffer &buf,
#ifdef WITH_GPU
                               GpuData &gpu_data,
#endif
                               const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores_sorted_asc, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores_sorted_desc, const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, bool run_on_gpu)
{
    EvalResult res;

    const size_t n_q1_chunks = buf.chunks_q1_n_pairs.size();
    const size_t n_q2_chunks = buf.chunks_q2_n_pairs.size();

    if (run_on_gpu)
    {
#ifdef WITH_GPU
        auto profiler_evaluate = std::make_unique<ScopedProfiler>("Evaluate solutions GPU      ");

        auto [done, q1_chunk, q2_chunk, solution_indices] = evaluate_gpu(gpu_data, buf.same_score_q1, buf.same_score_q2, ms_inst, reduce_dim, buf.chunks_q1_beg, buf.chunks_q1_n_pairs, buf.chunks_q2_beg, buf.chunks_q2_n_pairs, n_q1_chunks, n_q2_chunks, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets
#ifndef NDEBUG
                                                                         ,
                                                                         buf.n_pairs_q1,
                                                                         buf.n_pairs_q2
#endif
        );

        res.found = done;
        res.solution = solution_indices;
        res.i_q1_chunk = q1_chunk;
        res.i_q2_chunk = q2_chunk;

        profiler_evaluate.reset();
#else
        (void)n_q1_chunks;
        (void)n_q2_chunks;
        (void)set1_subsets;
        (void)set2_subsets_sorted_asc;
        (void)set3_subsets;
        (void)set4_subsets_sorted_desc;
        (void)subset_sum_1d;
        (void)offsets;

        printf("Error: GPU mode not available!\n\nAborting!\n");
        exit(1);
#endif
    }
    else
    {
        assert(n_q1_chunks == 1);
        assert(n_q2_chunks == 1);
        auto profiler_evaluate = std::make_unique<ScopedProfiler>("Evaluate solutions CPU      ");

        auto [done, solution_indices] = evaluate_cpu(set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc, buf.same_score_q1, buf.n_pairs_q1, buf.same_score_q2, buf.n_pairs_q2, ms_inst, reduce_dim, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets);

        res.found = done;
        res.solution = solution_indices;
        res.i_q1_chunk = 0;
        res.i_q2_chunk = 0;

        profiler_evaluate.reset();
    }

    buf.state = EVALUATED;

    return res;
}

template <typename T>
bool print_and_verify_solution(const PipelineBuffer &buf, const EvalResult &res, const MarkShareFeas &ms_inst, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, const std::vector<T> &asc_indices_set2_weights, const std::vector<T> &desc_indices_set4_weights, const std::vector<std::vector<size_t>> &set1_subsets, const std::vector<std::vector<size_t>> &set2_subsets_sorted_asc, const std::vector<std::vector<size_t>> &set3_subsets, const std::vector<std::vector<size_t>> &set4_subsets_sorted_desc, std::vector<T> &list1, std::vector<T> &list2, std::vector<T> &list3, std::vector<T> &list4, const std::string &instance_name

#ifndef NDEBUG
                               ,
                               size_t subset_sum_1d_rhs,
                               const std::vector<T> &set1_weights, const std::vector<T> &set2_weights_sorted_asc, const std::vector<T> &set3_weights, const std::vector<T> &set4_weights_sorted_desc
#endif
)
{
    /* Get the correct pairs. */
    size_t pos_q1 = buf.chunks_q1_beg[res.i_q1_chunk];
    size_t pos_q2 = buf.chunks_q2_beg[res.i_q2_chunk];

    while (res.solution.first >= buf.same_score_q1[pos_q1].pairs_offset + buf.same_score_q1[pos_q1].pairs_n_second)
        ++pos_q1;
    while (res.solution.second >= buf.same_score_q2[pos_q2].pairs_offset + buf.same_score_q2[pos_q2].pairs_n_second)
        ++pos_q2;

    assert(pos_q1 < buf.chunks_q1_beg[res.i_q1_chunk + 1]);
    assert(pos_q2 < buf.chunks_q2_beg[res.i_q2_chunk + 1]);
    assert(res.solution.first - buf.same_score_q1[pos_q1].pairs_offset < buf.same_score_q1[pos_q1].pairs_n_second);
    assert(res.solution.second - buf.same_score_q2[pos_q2].pairs_offset < buf.same_score_q2[pos_q2].pairs_n_second);

    const size_t pair_q1_second = buf.same_score_q1[pos_q1].pairs_second_beg + res.solution.first - buf.same_score_q1[pos_q1].pairs_offset;
    const size_t pair_q2_second = buf.same_score_q2[pos_q2].pairs_second_beg + res.solution.second - buf.same_score_q2[pos_q2].pairs_offset;
    std::pair<size_t, size_t> pair_q1 = {buf.same_score_q1[pos_q1].pairs_first, pair_q1_second};
    std::pair<size_t, size_t> pair_q2 = {buf.same_score_q2[pos_q2].pairs_first, pair_q2_second};

    std::vector<size_t> solution_1d(subset_sum_1d.size());

    assert(set1_weights[pair_q1.first] + set2_weights_sorted_asc[pair_q1.second] + set3_weights[pair_q2.first] + set4_weights_sorted_desc[pair_q2.second] == subset_sum_1d_rhs);

    const std::vector<const std::vector<size_t> *> vectors = {&set1_subsets[pair_q1.first], &set2_subsets_sorted_asc[pair_q1.second], &set3_subsets[pair_q2.first], &set4_subsets_sorted_desc[pair_q2.second]};

    size_t len;
    concat_vectors(solution_1d, len, vectors, offsets);

    /* We found a solution. Construct it, print it, and return. */
    if (!ms_inst.is_solution_feasible(solution_1d, len))
    {
        printf("Error, solution is not feasible!\n");
        exit(1);
    }

    printf("Found market share solution from SS-Algorithm!\n");
    print_four_list_solution(pair_q1.first, asc_indices_set2_weights[pair_q1.second], pair_q2.first, desc_indices_set4_weights[pair_q2.second], list1, list2, list3, list4);

    if (!instance_name.empty())
        write_four_list_solution_to_file(pair_q1.first, asc_indices_set2_weights[pair_q1.second], pair_q2.first, desc_indices_set4_weights[pair_q2.second], list1, list2, list3, list4, instance_name);
    return true;
}

template <typename T>
bool shroeppel_shamir_dim_reduced(const MarkShareFeas &ms_inst, bool run_on_gpu, const std::string &instance_name, size_t n_reduce_max)
{
    std::cout << "Running reduced dim shroeppel shamir" << std::endl;
    std::cout << "Running with " << omp_get_max_threads() << " threads" << std::endl;

    /* First, attempt some dimensionality reduction/perfect hashing.
     * We know that 0 <= A[i][j] <= 200 and thus sum_j A[i][j] <= n * 200.
     *
     * So, reserving n * 200 + 1 intervals for each of the entries i of a vector A.j we have an overlap free combination.
     * The amount of dimension we can remove this way is limited by the maximum value of T, the index type we are using for the 1-dimensional subset sum problem.
     */
    constexpr size_t max_coeff = 200;
    const size_t n_cols = ms_inst.n();

    const auto [max_reduce_dim, basis] = max_encodable_dimension<T>(max_coeff, n_cols);
    const size_t reduce_dim = std::max(size_t(1), std::min(std::min(ms_inst.m(), max_reduce_dim), n_reduce_max));
    const size_t leftover_dim = ms_inst.m() - reduce_dim;

    printf("Reducing %ld dimensions for Shroeppel-Shamir - leaving %ld for verification\n", reduce_dim, leftover_dim);

    const auto [subset_sum_1d, subset_sum_1d_rhs] = ms_inst.reduce_first_dimensions(reduce_dim, basis);

    const size_t split_index1 = subset_sum_1d.size() / 4;
    const size_t split_index2 = subset_sum_1d.size() / 2;
    const size_t split_index3 = 3 * subset_sum_1d.size() / 4;
    printf("Splitting sets into [0, %ld]; [%ld, %ld]; [%ld, %ld]; [%ld, %ld]\n", split_index1 - 1, split_index1, split_index2 - 1, split_index2, split_index3 - 1, split_index3, subset_sum_1d.size());

    auto profiler = std::make_unique<ScopedProfiler>("Setup time                  ");
    auto profilerTotal = std::make_unique<ScopedProfiler>("Solution time               ");

    std::vector<T> list1(subset_sum_1d.begin(), subset_sum_1d.begin() + split_index1);
    std::vector<T> list2(subset_sum_1d.begin() + split_index1, subset_sum_1d.begin() + split_index2);
    std::vector<T> list3(subset_sum_1d.begin() + split_index2, subset_sum_1d.begin() + split_index3);
    std::vector<T> list4(subset_sum_1d.begin() + split_index3, subset_sum_1d.end());
    assert(list1.size() + list2.size() + list3.size() + list4.size() == subset_sum_1d.size());

    const std::vector<size_t> offsets = {0, list1.size(), list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};
    const std::vector<size_t> offsetsQ1 = {0, list1.size()};
    const std::vector<size_t> offsetsQ2 = {list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};

    auto [set1_weights, set1_subsets] = generate_subsets(list1);
    auto [set2_weights, set2_subsets] = generate_subsets(list2);
    auto [set3_weights, set3_subsets] = generate_subsets(list3);
    auto [set4_weights, set4_subsets] = generate_subsets(list4);

    /* Sort set2_weights ascending, set4_weights descending. */
    auto asc_indices_set2_weights = sort_indices(set2_weights, true);
    auto desc_indices_set4_weights = sort_indices(set4_weights, false);

    const auto set2_weights_sorted_asc = apply_permutation(set2_weights, asc_indices_set2_weights);
    const auto set2_subsets_sorted_asc = apply_permutation(set2_subsets, asc_indices_set2_weights);

    const auto set4_weights_sorted_desc = apply_permutation(set4_weights, desc_indices_set4_weights);
    const auto set4_subsets_sorted_desc = apply_permutation(set4_subsets, desc_indices_set4_weights);

    std::vector<size_t> set1_scores(set1_subsets.size() * leftover_dim);
    std::vector<size_t> set2_scores_sorted_asc(set2_subsets.size() * leftover_dim);
    std::vector<size_t> set3_scores(set3_subsets.size() * leftover_dim);
    std::vector<size_t> set4_scores_sorted_desc(set4_subsets.size() * leftover_dim);

    compute_scores_cpu(ms_inst, set1_scores, set1_subsets, offsets[0], reduce_dim);
    compute_scores_cpu(ms_inst, set2_scores_sorted_asc, set2_subsets_sorted_asc, offsets[1], reduce_dim);
    compute_scores_cpu(ms_inst, set3_scores, set3_subsets, offsets[2], reduce_dim);
    compute_scores_cpu(ms_inst, set4_scores_sorted_desc, set4_subsets_sorted_desc, offsets[3], reduce_dim);

#ifdef WITH_GPU
    GpuData gpu_data(ms_inst, set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc);
#endif

    /* Create the priority queues q1 consisting of pairs {(i, 0) | i \in set1_weights} and q2 consisting of {(i, 0) | i \in set3_weights}. The priority/score for a pair (i, j)
     * is given set1_weights[i] + set2_weights[j] if the pair is in q1 and set3_weights[i] + set4_weights[j] if the pair is in q2. */

    /* Compare returns true if the first argument comes BEFORE the second argument. Since however the priority queue outputs the largest element first,
     * we have to flip the > signs. */
    auto min_cmp = [&](std::pair<T, T> a1, std::pair<T, T> a2) -> bool
    {
        return set1_weights[a1.first] + set2_weights_sorted_asc[a1.second] > set1_weights[a2.first] + set2_weights_sorted_asc[a2.second];
    };

    auto max_cmp = [&](std::pair<T, T> a1, std::pair<T, T> a2) -> bool
    {
        return set3_weights[a1.first] + set4_weights_sorted_desc[a1.second] < set3_weights[a2.first] + set4_weights_sorted_desc[a2.second];
    };

    /* Vectors used to count the number of elements extracted from q1/q2 with the same solution value. */
    /* Each tuple {a, b, c, d} will describe the range of pairs <a, b> ... <a, b + c - 1>; d denotes the offset of the pairs within a list of all pairs. */
    std::vector<std::pair<size_t, size_t>> heap1;
    heap1.reserve(set1_weights.size());
    std::vector<std::pair<size_t, size_t>> heap2;
    heap2.reserve(set3_weights.size());

    // TODO: the initial insert can likely be improved by simple sorting.
    for (size_t i = 0; i < set1_weights.size(); ++i)
    {
        /* If already the sum of these 2 elements is greater than the right hand side we can skip them. Subsequent combinations (e.g. with higher pos_subset2)
         * will only be even larger. */
        if (set1_weights[i] + set2_weights_sorted_asc[0] <= subset_sum_1d_rhs)
            heap1.emplace_back(i, 0);
    }

    for (size_t i = 0; i < set3_weights.size(); ++i)
        heap2.emplace_back(i, 0);

    std::make_heap(heap1.begin(), heap1.end(), min_cmp);
    std::make_heap(heap2.begin(), heap2.end(), max_cmp);

    printf("Running the search loop\n\n");

    profiler = std::make_unique<ScopedProfiler>("List traversal              ");

    PipelineBuffer buffers[2];
    std::future<EvalResult> eval_future[2];

    buffers[0].same_score_q1.reserve(100000);
    buffers[0].same_score_q2.reserve(100000);
    buffers[1].same_score_q1.reserve(100000);
    buffers[1].same_score_q2.reserve(100000);

    buffers[0].state = EMPTY;
    buffers[1].state = EMPTY;

    size_t curr = 0;
    size_t next = 1;
    size_t i_iter_checking = 0;

    while (!heap1.empty() && !heap2.empty())
    {
        /* score_pair1 is the currently lowest score in {set1_weights, set2_weights} we are still considering */
        const T score_pair1 = set1_weights[heap1.front().first] + set2_weights_sorted_asc[heap1.front().second];
        /* score_pair2 is the currently highest score in {set3_weights, set4_weights} we are still considering */
        const T score_pair2 = set3_weights[heap2.front().first] + set4_weights_sorted_desc[heap2.front().second];

        const T score = score_pair1 + score_pair2;

        if (score == subset_sum_1d_rhs)
        {
            /* Extract all tuples from both lists with equal scores. Potentially, we subdivide extracted tuples into chunks to not overflow the GPU memory. We remember the start and number of pairs in each chunk. */
            auto &buf_curr = buffers[curr];
            auto &buf_next = buffers[next];
            assert(buf_curr.state == EMPTY);
            buf_curr.state = EXTRACTING;

            /* Reset the buffer. */
            /* Clear vectors but keep their old capacity. */
            buf_curr.same_score_q1.clear();
            buf_curr.same_score_q2.clear();

            buf_curr.chunks_q1_beg.clear();
            buf_curr.chunks_q2_beg.clear();

            buf_curr.chunks_q1_n_pairs.clear();
            buf_curr.chunks_q2_n_pairs.clear();

            buf_curr.chunks_q1_beg.push_back(0);
            buf_curr.chunks_q2_beg.push_back(0);

            ++i_iter_checking;

            buf_curr.n_pairs_q1 = 0;
            buf_curr.n_pairs_q2 = 0;

            auto profiler_cand_extraction = std::make_unique<ScopedProfiler>("Candidate extraction        ");

            /* In CPU parallel, extract equal tuples from each heap. */
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    extract_pairs_from_heap<true, T>(buf_curr.same_score_q1, heap1, score_pair1, buf_curr.chunks_q1_beg, buf_curr.chunks_q1_n_pairs, buf_curr.n_pairs_q1, set1_weights, set2_weights, set2_weights_sorted_asc, min_cmp);
                }
#pragma omp section
                {
                    extract_pairs_from_heap<false, T>(buf_curr.same_score_q2, heap2, score_pair2, buf_curr.chunks_q2_beg, buf_curr.chunks_q2_n_pairs, buf_curr.n_pairs_q2, set3_weights, set4_weights, set4_weights_sorted_desc, max_cmp);
                }
            }
            buf_curr.state = READY_FOR_EVAL;

            profiler_cand_extraction.reset();
            print_info_line(
#ifdef WITH_GPU
                gpu_data,
                run_on_gpu,
#endif
                i_iter_checking, profilerTotal->elapsed(), score_pair1, score_pair2, buf_curr.n_pairs_q1, buf_curr.n_pairs_q2);

            /* Before submitting the current buffer, wait until the last one is finished. */
            if (buf_next.state != EMPTY)
            {
                eval_future[next].wait();
                const auto &result = eval_future[next].get();
                assert(buf_next.state == EVALUATED);

                /* Check the result - break if we are done. */
                if (result.found)
                {
                    return print_and_verify_solution(buf_next, result, ms_inst, subset_sum_1d, offsets, asc_indices_set2_weights, desc_indices_set4_weights, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, list1, list2, list3, list4, instance_name
#ifndef NDEBUG
                                                     ,
                                                     subset_sum_1d_rhs,
                                                     set1_weights,
                                                     set2_weights_sorted_asc, set3_weights, set4_weights_sorted_desc
#endif
                    );
                }

                buf_next.state = EMPTY;
            }
            assert(buf_next.state == EMPTY);

            buf_curr.state = EVALUATING;
            /* Launch evaluation in background thread. */
            eval_future[curr] = std::async(std::launch::async, [&]()
                                           { return evaluate_gpu_or_cpu(buf_curr,
#ifdef WITH_GPU
                                                                        gpu_data,
#endif
                                                                        ms_inst, reduce_dim, set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, run_on_gpu); });

            /* Switch buffer. */
            curr = next;
            next = 1 - curr;
        }
        else if (score < subset_sum_1d_rhs)
        {
            const auto pair1 = heap1.front();
            size_t pos_set2_weights = pair1.second;

            std::pop_heap(heap1.begin(), heap1.end(), min_cmp);
            heap1.pop_back();

            ++pos_set2_weights;

            while (pos_set2_weights + 1 < set2_weights.size() && (set2_weights_sorted_asc[pos_set2_weights] == set2_weights_sorted_asc[pair1.second] || (set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] + score_pair2) < subset_sum_1d_rhs))
                ++pos_set2_weights;

            /* Again, the element in q1 can only increase (or stay equal). So ignore elements that are already too big. */
            if (pos_set2_weights < set2_weights.size() && set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] <= subset_sum_1d_rhs)
            {
                heap1.emplace_back(pair1.first, pos_set2_weights);
                std::push_heap(heap1.begin(), heap1.end(), min_cmp);
            }
        }
        else if (score > subset_sum_1d_rhs)
        {
            const auto pair2 = heap2.front();
            size_t pos_set4_weights = pair2.second;

            std::pop_heap(heap2.begin(), heap2.end(), max_cmp);
            heap2.pop_back();

            ++pos_set4_weights;

            /* Skip all entries in set4_weights until we find a smaller one. */
            while (pos_set4_weights + 1 < set4_weights.size() && (set4_weights_sorted_desc[pos_set4_weights] == set4_weights_sorted_desc[pair2.second] || (score_pair1 + set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights]) > subset_sum_1d_rhs))
                ++pos_set4_weights;

            if (pos_set4_weights < set4_weights.size() && set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights] <= subset_sum_1d_rhs)
            {
                heap2.emplace_back(pair2.first, pos_set4_weights);
                std::push_heap(heap2.begin(), heap2.end(), max_cmp);
            }
        }
    }

    profiler.reset();

    return false;
}

std::string get_filename_without_extension(const std::string &filePath)
{
    // Find the last path separator '/'
    size_t lastSlash = filePath.find_last_of('/');
    size_t start = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;

    // Find the last dot after the last slash
    size_t lastDot = filePath.find_last_of('.');
    size_t end = (lastDot == std::string::npos || lastDot < start) ? filePath.length() : lastDot;

    // Extract the filename without extension
    return filePath.substr(start, end - start);
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("markshare");

    std::string path = "";
    size_t n_iter = 1;
    size_t seed = 2025;
    size_t m = 5;
    size_t n = 0;
    size_t n_reduce = 0;
    size_t k = 100;

    program.add_argument("-m", "--m")
        .store_into(m)
        .help("Number of rows of the markshare problem.");

    program.add_argument("-n", "--n")
        .store_into(n)
        .help("Number of columns of the markshare problem. Set to (m - 1) * 10 if not given. ");

    program.add_argument("-k", "--k")
        .store_into(k)
        .help("Coefficients are generated in the range [0, k).")
        .default_value(100);

    program.add_argument("--reduce")
        .store_into(n_reduce)
        .help("Number of rows (max) to be reduced. Only effective if --reduced is set. ")
        .default_value(0);

    program.add_argument("-s", "--seed")
        .store_into(seed)
        .help("Random seed for instance generation.")
        .default_value(2025);

    program.add_argument("-i", "--iter")
        .store_into(n_iter)
        .help("Number of problems to solve. Seed for problem of iteration i (starting from 0) is seed + i.")
        .default_value(1);

    program.add_argument("--gpu")
        .help("Run validation on GPU")
        .flag();

    program.add_argument("-f", "--file")
        .store_into(path)
        .help("Supply instance path to read instance from. Overrides '-m', '-n', '-k', and '-i'");

    program.add_argument("--max_pairs")
        .store_into(max_pairs_per_chunk)
        .help("Maximum number of pairs to be evaluated on the GPU simultaneously. If GPU runs OOM, reduce this number.")
        .default_value(3500000000);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    /* Adjust n. */
    if (n == 0)
        n = (m - 1) * 10;

    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter)
    {
        std::string instance_name{};
        const size_t seed_iter = seed + i_iter;
        MarkShareFeas instance;

        /* Generate/read instance. For now, random instances. */
        if (!path.empty())
        {
            instance_name = get_filename_without_extension(path);
            printf("Reading instance from file %s; instance_name %s\n", path.c_str(), instance_name.c_str());
            instance = MarkShareFeas(path);
        }
        else
        {
            instance = MarkShareFeas(m, n, k, seed_iter);
            instance_name = "markshare_m_" + std::to_string(m) + "_n_" + std::to_string(n) + "_seed_" + std::to_string(seed_iter);
            instance.write_as_prb(instance_name + ".prb");
        }

        printf("Running markshare: m=%ld, n=%ld, seed=%ld, iter=%ld, nthread=%d\n", instance.m(), instance.n(), seed_iter, i_iter, omp_get_max_threads());
        instance.print();

        /* Solve the instance using one of the available algorithms. */

        /* Create the one dimensional subset sum problem. */
        const bool on_gpu = (program["--gpu"] == true);
        const bool found = shroeppel_shamir_dim_reduced<uint64_t>(instance, on_gpu, instance_name, n_reduce);

        if (found)
            printf("Found feasible solution!\n");
        else
            printf("Instance was infeasible .. \n");
    }

    ScopedProfiler::report();

    return 0;
}