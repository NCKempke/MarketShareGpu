#pragma once

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

/* A market share feasibility instance. */
class MarkShareFeas
{
public:
    MarkShareFeas() = default;
    ~MarkShareFeas() = default;
    MarkShareFeas(MarkShareFeas &&) noexcept = default;
    MarkShareFeas &operator=(MarkShareFeas &&) = default;
    MarkShareFeas &operator=(const MarkShareFeas &) = default;
    MarkShareFeas(const MarkShareFeas &) = default;

    MarkShareFeas(size_t m, size_t n, std::vector<size_t> &&A, std::vector<size_t> &&b) : matrix(std::move(A)), rhs(std::move(b)), m_rows(m), n_cols(n)
    {
        assert(matrix.size() == m * n);
        assert(rhs.size() == m);
    }

    /* Generates a market share feasibility instance (or a multi-dimensional subset sum problem)
     *   Ax = b
     * with A \in \Z^{mxn}, a_ij \in \{0,..,99\}, n, m \in \N, b_i = |_ 1/2 \sum^n_j=1 a_ij _|
     *
     * See
     * Gérard Cornuéjols, Milind Dawande, (1999) A Class of Hard Small 0-1 Programs. INFORMS Journal on Computing 11(2):205-210, https://doi.org/10.1287/ijoc.11.2.205.
     */
    MarkShareFeas(size_t m, size_t n, size_t k, size_t seed = 0) : m_rows{m}, n_cols{n}
    {
        matrix.reserve(m_rows * n_cols);
        rhs.reserve(m_rows);

        constexpr size_t lower = 0;
        size_t upper = k - 1;

        std::mt19937 generator(seed);

        /* Generate the instance truly random if no seed was given. */
        if (seed == 0)
        {
            std::random_device rd;
            generator.seed(rd());
        }

        std::uniform_int_distribution<size_t> distribution(lower, upper);

        /* Fill matrix A and right hand side b. */
        for (size_t m = 0; m < m_rows; ++m)
        {
            size_t row_sum = 0;

            for (size_t n = 0; n < n_cols; ++n)
            {
                size_t next = distribution(generator);
                matrix.push_back(next);
                row_sum += next;
            }

            rhs.push_back(std::floor(0.5 * row_sum));
        }
    }

    /* Generates a market share feasibility instance with n = 10 * (m - 1). */
    MarkShareFeas(size_t m, size_t seed = 0) : MarkShareFeas(m, 10 * (m - 1), seed) {};

    static void skip_comments_and_empty_lines(std::istream& in)
    {
        while (true)
        {
            in >> std::ws; // skip whitespace, including empty lines

            if (in.peek() == '#')
            {
                std::string ignored;
                std::getline(in, ignored); // consume the full comment line
                continue;
            }

            break;
        }
    }

    /* Read a MarkShareInstance from file. The format supported is
     * +++++++++++++++++++++++++++++++++++++++
     * m n
     * a11 a12 a13 ... a1n b1
     * a21 a22 ...     a2n b2
     *  .               .  .
     *  .               .  .
     *  .               .  .
     * am1 am2 am3     amn bm
     * +++++++++++++++++++++++++++++++++++++++
     *
     * Lines starting with '#' are treated as comments.
     */
    MarkShareFeas(const std::string instance_path)
    {
        std::ifstream file(instance_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file");
        }

        /* Read the first line to get m and n. */
        skip_comments_and_empty_lines(file);
        file >> m_rows >> n_cols;

        matrix.reserve(m_rows * n_cols);
        rhs.reserve(m_rows);

        /* Read the matrix rows and right hand side. */
        for (size_t i_row = 0; i_row < m_rows; ++i_row)
        {
            skip_comments_and_empty_lines(file);

            for (size_t j_col = 0; j_col < n_cols; ++j_col)
            {
                size_t value;
                file >> value;
                matrix.push_back(value);
            }

            size_t rhs_val;
            file >> rhs_val;

            rhs.push_back(rhs_val);
        }
    };

    const std::vector<size_t> &A() const
    {
        return matrix;
    }

    const std::vector<size_t> &b() const
    {
        return rhs;
    }

    size_t m() const
    {
        return m_rows;
    }

    size_t n() const
    {
        return n_cols;
    }

    template <typename T>
    std::pair<std::vector<T>, T> reduce_first_dimensions(size_t n_dim, T basis) const
    {
        std::vector<T> reduced_dims(n_cols, 0);
        T basis_to_power = 1;
        T reduced_rhs = 0;

        assert(n_dim <= m_rows);

        for (size_t i_dim = 0; i_dim < n_dim; ++i_dim)
        {
            for (size_t j_col = 0; j_col < n_cols; ++j_col)
                reduced_dims[j_col] += basis_to_power * matrix[i_dim * n_cols + j_col];
            reduced_rhs += basis_to_power * rhs[i_dim];

            basis_to_power *= basis;
        }

        return {reduced_dims, reduced_rhs};
    }

    bool check_sum_feas(const size_t *values1, const size_t *values2) const
    {
        assert(values1[0] + values2[0] == rhs[0]);

        for (size_t row = 1; row < m_rows; ++row)
        {
            if (values1[row] + values2[row] != rhs[row])
                return false;
        }
        return true;
    }

    void compute_value(const std::vector<size_t> &indices, size_t col_offset, size_t row_offset, size_t *values) const
    {
        const size_t m_rows_left = m_rows - row_offset;

        for (size_t row = 0; row < m_rows_left; ++row)
        {
            size_t row_prob = row + row_offset;
            size_t val = 0;
            for (size_t i_idx = 0; i_idx < indices.size(); ++i_idx)
            {
                size_t j_col = indices[i_idx] + col_offset;
                val += matrix[row_prob * n_cols + j_col];
            }

            values[row] = val;
        }
    }

    bool is_solution_feasible(const std::vector<size_t> &indices, size_t len_indices) const
    {
        for (size_t row = 0; row < m_rows; ++row)
        {
            size_t val = 0;
            for (size_t i_index = 0; i_index < len_indices; ++i_index)
            {
                size_t i = indices[i_index];
                val += matrix[row * n_cols + i];
                if (val > rhs[row])
                {
                    assert(row != 0);
                    return false;
                }
            }

            if (val != rhs[row])
            {
                assert(row != 0);
                return false;
            }
        }
        return true;
    }

    void print() const
    {
        /* Find max width needed. */
        size_t max_width_mat = 0;
        size_t max_width_rhs = 0;

        auto update_width = [](double value, size_t& max_width)
        {
            std::ostringstream oss;
            oss << value;
            size_t width = oss.str().length();
            if (width > max_width)
                max_width = width;
        };

        for (size_t row = 0; row < m_rows; ++row)
        {
            for (size_t col = 0; col < n_cols; ++col)
            {
                update_width(matrix[row * n_cols + col], max_width_mat);
            }
            update_width(rhs[row], max_width_rhs);
        }

        const size_t padding = 1;
        max_width_mat += padding;
        max_width_rhs += padding;

        /* Print. */
        std::cout << "[\n";
        for (size_t row = 0; row < m_rows; ++row)
        {
            std::cout << " [";
            for (size_t col = 0; col < n_cols; ++col)
            {
                std::cout << std::setw(max_width_mat) << matrix[row * n_cols + col];
            }
            std::cout << " |" << std::setw(max_width_rhs) << rhs[row] << " ]\n";
        }
        std::cout << "]\n";
    }

    void write_as_prb(const std::string &name) const
    {
        std::cout << "Storing mark share instance as " << name << "\n";
        std::ofstream file(name);

        file << "[\n"
             << "]\n";
        file << "\n";

        file << "[\n";
        /* Write the matrix. */
        for (size_t row = 0; row < m_rows; ++row)
        {
            file << "[";
            for (size_t col = 0; col < n_cols; ++col)
            {
                file << " " << matrix[row * n_cols + col];
            }
            file << " ]\n";
        }
        file << "]\n";

        file << "[ ]\n";
        file << "\n";
        file << "[ ]\n";
        file << "\n";

        file << "[";
        /* Write the right hand side. */
        for (size_t row = 0; row < m_rows; ++row)
        {
            file << " " << rhs[row];
        }
        file << " ]\n";

        file << "\n";

        file << "[";
        /* Write 1s for each column. */
        for (size_t col = 0; col < n_cols; ++col)
        {
            file << " 1";
        }
        file << "]\n";

        file << "\n";
        file << "0";
        file << "\n";
    }

private:
    std::vector<size_t> matrix;
    std::vector<size_t> rhs;
    size_t m_rows;
    size_t n_cols;
};